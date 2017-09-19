import pandas as pd
from scipy.io.arff import loadarff
from scipy.stats import randint as sp_randint
import scipy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, silhouette_samples, silhouette_score
from scipy.stats import mode
from collections import Counter

import warnings

dataset_path = "./datasets/"
wisdm_v1_dataset_path = dataset_path + "WISDM_v1/"
wisdm_v2_dataset_path = dataset_path + "WISDM_v2/"

wisdm_transformed_v1 = "WISDM_ar_v1.1_transformed_FIXED.arff"
wisdm_transformed_v2 = "WISDM_at_v2.0_transformed_FIXED.arff"

WISDM_DIR = wisdm_v1_dataset_path # switch this at anytime to fix
WISDM_TRANSFORMED = wisdm_transformed_v1

data_df = None
user_ids = None

import time                                                

def timeit(method):
	number_of_cores = 32

	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		print("%s took %s minutes" % (method.__name__, (te-ts)/60.))

		return result

	return timed

def make_labels_compatible(data_df):
	class_labels = data_df['class'].unique()

	if b'LyingDown' in class_labels:
		# remove rows with "LyingDown" as class
		new_data_df = data_df[data_df['class'] != b"LyingDown"]
		return new_data_df
	elif (b'Upstairs' in class_labels) or (b'Downstairs' in class_labels):
		new_data_df = data_df.copy()
		new_data_df['class'] = data_df['class'].replace(to_replace=[b'Upstairs', b'Downstairs'], value=[b'Stairs', b'Stairs'])
		return new_data_df

def get_features():
	path = WISDM_DIR + WISDM_TRANSFORMED
	data, metadata = loadarff(path)
	data_df = pd.DataFrame.from_records(data, columns=metadata.names())
	data_df.columns = [col.replace('"', '') for col in data_df.columns]
	data_df["user"] = [x.decode("utf-8") for x in data_df["user"]]
	return data_df

def get_data():
	data_df = get_features()
	if 'UNIQUE_ID' in data_df.columns:
		del data_df['UNIQUE_ID']
	user_ids = data_df['user'].unique()
	return data_df, user_ids

def set_data(version='1', make_compatible=False):
	global data_df
	global user_ids
	global WISDM_DIR
	global WISDM_TRANSFORMED

	if (version is '1') or (version is '1.1'):
		WISDM_DIR = wisdm_v1_dataset_path
		WISDM_TRANSFORMED = wisdm_transformed_v1
	elif version is '2':
		WISDM_DIR = wisdm_v2_dataset_path
		WISDM_TRANSFORMED = wisdm_transformed_v2
	
	data_df, user_ids = get_data()
	if make_compatible:
		data_df = make_labels_compatible(data_df)

def get_demographics_description():
	with open(WISDM_DIR+"WISDM_at_v2.0_demographics_about.txt") as fIn:
		return fIn.read()

def get_demographics():
	demo_df = pd.read_csv("WISDM_at_v2.0/WISDM_at_v2.0_demographics.txt")
	demo_df.columns = ['uid', 'height', 'sex','age','weight','leg_injury']
	return demo_df

def get_random_user_set(n):
	random_indeces = np.random.choice(len(user_ids), n, replace=False)
	random_ids = [user_ids[r] for r in random_indeces]
	df_set = data_df[data_df['user'].isin(random_ids)]
	return df_set

def get_user_set(user_id):
	user_set = data_df[data_df['user'] == user_id]
	return user_set

# Utility function to report best scores
def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("\tModel with rank: {0}".format(i))
			print("\tMean validation score: {0:.3f} (std: {1:.3f})".format(
					results['mean_test_score'][candidate],
					results['std_test_score'][candidate]))
			print("\tParameters: {0}".format(results['params'][candidate]))
			print("")

# Data cleaning
def remove_all_nan(df):
		df = df.replace([np.inf, -np.inf], np.nan)
		null_rows = df.isnull().any(axis=1)
		
		return df.drop(df.index[null_rows.tolist()])



# Classifier getters
def rfc_optimized(param_dist={"max_depth": [3, None],
											"max_features": sp_randint(1, 11),
											"min_samples_split": sp_randint(2, 11),
											"min_samples_leaf": sp_randint(1, 11),
											"bootstrap": [True, False],
											"criterion": ["gini", "entropy"]},
										param_grid=None, n_estimators=20, n_iter_search=30, n_jobs=1):
		clf = RandomForestClassifier(n_estimators=n_estimators)
		if not param_grid:
				random_search = RandomizedSearchCV(clf, param_distributions=param_dist,n_iter=n_iter_search, n_jobs=n_jobs)
				return random_search
		else:
				grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=n_jobs)
				return grid_search

def knn_optimized(param_dist={"n_neighbors":sp_randint(1,30)}, param_grid=None, n_iter_search=30, n_jobs=1):
		clf = KNeighborsClassifier()
		if not param_grid:
				random_search = RandomizedSearchCV(clf, param_distributions=param_dist,n_iter=n_iter_search, n_jobs=n_jobs)
				return random_search
		else:
				grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=n_jobs)
				return grid_search

def svc_optimized(param_dist=[{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
															'kernel': ['rbf'], 'class_weight':['balanced', None]},
														{'kernel': 'linear', 'C': scipy.stats.expon(scale=100)}],
									param_grid=None, n_iter_search=30, n_jobs=1):
		clf = SVC(max_iter=1e7)
		if not param_grid:
				random_search = RandomizedSearchCV(clf, param_distributions=param_dist,n_iter=n_iter_search, n_jobs=n_jobs)
				return random_search
		else:
				grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=n_jobs, verbose=3)
				return grid_search

def weka_RF():
		# trees.RandomForest '-P 100 -I 100 -num-slots 1 
		#                     -K 0 -M 1.0 -V 0.001 -S 1' 1116839470751428698a
		clf = RandomForestClassifier(n_estimators=100, max_features="log2",
																 min_samples_leaf=1)
		return clf

##
def personal_model(active_features, active_labels, test_features=None, test_labels=None):
	scaler = StandardScaler().fit(active_features)
	scaled_train_x = scaler.transform(active_features)

	rfc_clf = weka_RF()
	rfc_clf.fit(scaled_train_x, active_labels)

	if test_features is None:
		return rfc_clf
	
	scaled_test_x = scaler.transform(test_features)
	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)
	return score

def impersonal_model(impersonal_features, impersonal_labels, test_features=None, test_labels=None):
	scaler = StandardScaler().fit(impersonal_features)
	scaled_train_x = scaler.transform(impersonal_features)
	
	rfc_clf = weka_RF()
	rfc_clf.fit(scaled_train_x, impersonal_labels)
	
	if test_features is None:
		return rfc_clf
	scaled_test_x = scaler.transform(test_features)

	
	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)

	return score

def impersonal_plus_personal_model(personal_features, personal_labels,
								  impersonal_features, impersonal_labels,
								  test_features=None, test_labels=None):
	personal_plus_impersonal_features = np.vstack((personal_features, impersonal_features))
	personal_plus_impersonal_labels = np.hstack((personal_labels, impersonal_labels))

	scaler = StandardScaler().fit(personal_plus_impersonal_features)
	scaled_train_x = scaler.transform(personal_plus_impersonal_features)

	rfc_clf = weka_RF()
	rfc_clf.fit(scaled_train_x, personal_plus_impersonal_labels)

	if test_features is None:
		return rfc_clf
	scaled_test_x = scaler.transform(test_features)

	
	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)
	return score

def cluster_plus_personal_model(personal_features, personal_labels,
								  impersonal_features, impersonal_labels,
								  KM, clusters,
								  test_features=None, test_labels=None):
	cluster_predictions = KM.predict(personal_features)
	closest_cluster = mode(cluster_predictions).mode[0]

	cluster_data_indeces = [i for i in range(len(clusters)) if clusters[i] == closest_cluster]
	cluster_features = impersonal_features[cluster_data_indeces]
	cluster_labels = impersonal_labels[cluster_data_indeces]

	training_features = np.vstack((personal_features, cluster_features))
	training_labels = np.hstack((personal_labels, cluster_labels))

	scaler = StandardScaler().fit(training_features)
	scaled_train_x = scaler.transform(training_features)


	rfc_clf = weka_RF()
	rfc_clf.fit(scaled_train_x, training_labels)

	if test_features is None:
		return rfc_clf
   
	scaled_test_x = scaler.transform(test_features)

	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)
	return score

def garcia_ceja_model(active_features, active_labels, \
						impersonal_features, impersonal_labels, \
						test_features=None, test_labels=None,
						k_upper_bound=7):
	training_features = None
	training_labels = None
	sample_weights = None

	class_labels = set(impersonal_labels)
	r = 0.2
	personal_weight = (1 - r) ** len(active_features)

	for cl in class_labels:
		active_cl_indeces = [ind for ind, pf in enumerate(active_features) if active_labels[ind] == cl]
		active_cl_features = active_features[active_cl_indeces]

		impersonal_cl_indeces = [ind for ind, impf in enumerate(impersonal_features) if impersonal_labels[ind] == cl]
		impersonal_cl_features = impersonal_features[impersonal_cl_indeces]

		if len(active_cl_indeces) == 0:
			#training_features = np.vstack((training_features, impersonal_cl_features))
			#training_labels = np.hstack((training_labels, np.full((len(impersonal_cl_features),), cl)))
			#sample_weights = np.hstack((sample_weights, np.full((len(impersonal_cl_features),), (1-personal_weight))))
			continue

		#print("active_cl_features.shape : (%s,%s)" % active_cl_features.shape)
		#print("impersonal_cl_features.shape : (%s, %s)" % impersonal_features.shape)
		cl_features = np.vstack((active_cl_features, impersonal_cl_features))

		# cluster combined data
		KM_objs = []
		KM_silhouette_scores = []
		for k in range(2, k_upper_bound):
			KM = KMeans(n_clusters=k)
			KM.fit(cl_features)
			cluster_labels = KM.predict(cl_features)
			silhouette_avg = silhouette_score(cl_features, cluster_labels)
			#print("k=%s, silhouette score=%s" % (k, silhouette_avg))
			KM_silhouette_scores.append(silhouette_avg)
			KM_objs.append(KM)

		best_k = np.argmax(KM_silhouette_scores)
		KM = KM_objs[best_k]

		# predict cluster for personal features
		predictions = KM.predict(active_cl_features)
		best_cluster_label = mode(predictions).mode[0]
		
		# add the personal data
		if training_features is None:
			training_features = active_cl_features
			training_labels = np.full((len(active_cl_features), ), cl)
			sample_weights = np.full((len(active_cl_features),), personal_weight)
		else:
			training_features = np.vstack((training_features, active_cl_features))
			training_labels = np.hstack((training_labels, np.full((len(active_cl_features),), cl)))
			sample_weights = np.hstack((sample_weights, np.full((len(active_cl_features),), personal_weight)))

		# add the impersonal data
		cluster_predictions = KM.predict(impersonal_features)
		cluster_impersonal_features = [impersonal_features[ind] for ind, pred in enumerate(cluster_predictions) if pred == best_cluster_label]
		training_features = np.vstack((training_features, cluster_impersonal_features))
		training_labels = np.hstack((training_labels, np.full((len(cluster_impersonal_features),), cl) ))
		sample_weights = np.hstack((sample_weights, np.full((len(cluster_impersonal_features),), (1-personal_weight))))

	# train the model
	hybrid_clf = weka_RF()
	hybrid_scaler = StandardScaler().fit(training_features)
	scaled_training_features = hybrid_scaler.transform(training_features)
	hybrid_clf.fit(scaled_training_features, training_labels, sample_weights)
	
	if test_features is None:
		return hybrid_clf

	# make predictions
	scaled_test_features = hybrid_scaler.transform(test_features)
	predictions = hybrid_clf.predict(scaled_test_features)
	score = accuracy_score(test_labels, predictions)
	return score

# uncertainty sampling methods
def least_confident_active_sampling(all_personal_features, model, number_of_samples):
	'''returns an array of length, number_of_samples, representing which indeces of the personal features to sample'''
	if number_of_samples > len(all_personal_features):
		raise ValueError('The number of personal samples provided (%s) is less than the number of least-certain samples requested(%s)' % (len(all_personal_features), number_of_samples))

	probabilities = model.predict_proba(all_personal_features)
	uniform_prob = 1. / len(model.classes_)
	diffs_from_uniform = []
	
	for ind, prob in enumerate(probabilities):
		max_prob = np.max(prob)
		diffs_from_uniform.append(max_prob - uniform_prob)

	least_certain_indeces = np.argsort(diffs_from_uniform)[:number_of_samples]
	return least_certain_indeces

def margin_active_sampling(all_personal_features, model, margin_size):
	probabilities = model.predict_proba(all_personal_features)
	uniform_prob = 1. / len(model.classes_)

	active_sample_indeces = []
	for ind, prob in enumerate(probabilities):
		sorted_probs = np.argsort(prob)
		max_prob = prob[sorted_probs[-1]]
		second_max_prob = prob[sorted_probs[-2]]
		if max_prob - second_max_prob < margin_size:
			active_sample_indeces.append(ind)

	return active_sample_indeces

def entropy_active_sampling(all_personal_features, all_personal_labels,
								  impersonal_features, impersonal_labels, margin_size):
	pass

@timeit
def sample_experiments(user_id, k_run, \
					  impersonal_features, impersonal_labels, \
					  potential_active_features, potential_active_labels, \
					  test_features, test_labels, \
					  training_sizes, \
					  random_sample_iterations, \
					  impersonal_model=None, impersonal_scaler=None, \
					  KM=None, clusters=None):
	# iteratively increase the number of samples we use
	rows = []

	for ts in training_sizes:
		# initialize score holders
		random_personal_scores = []
		random_personal_plus_all_scores = []
		random_personal_plus_cluster_scores = []
		#random_gc_scores = []

		# run impersonal model
		impersonal_scaled_test_x = impersonal_scaler.transform(test_features)
		impersonal_model_score = accuracy_score(test_labels, impersonal_model.predict(impersonal_scaled_test_x))

		for run in range(random_sample_iterations):
			# get random samples
			try:
				random_active_indeces = np.random.choice(len(potential_active_features), ts, replace=False)
			except ValueError as ve:
				if """Cannot take a larger sample than population when 'replace=False'""" in ve.args[0]:
					continue
			sampled_active_features = potential_active_features[random_active_indeces]
			sampled_active_labels = potential_active_labels[random_active_indeces]

			# run personal model
			random_personal_score = personal_model(sampled_active_features, sampled_active_labels, test_features=test_features, test_labels=test_labels)
			random_personal_scores.append(random_personal_score)

			# run personal + universal
			random_personal_plus_all_score = impersonal_plus_personal_model(sampled_active_features, sampled_active_labels,
																	impersonal_features, impersonal_labels,
																	test_features=test_features, test_labels=test_labels)
			random_personal_plus_all_scores.append(random_personal_plus_all_score)

			# run personal + cluster
			random_personal_plus_cluster_score = cluster_plus_personal_model(sampled_active_features, sampled_active_labels,
																	impersonal_features, impersonal_labels,
																	KM, clusters,
																	test_features=test_features, test_labels=test_labels)

			random_personal_plus_cluster_scores.append(random_personal_plus_cluster_score)

			# run garcia-ceja personalization approach
			#random_gc_score = garcia_ceja_model(sampled_active_features, sampled_active_labels,
			#							impersonal_features, impersonal_labels,
			#								test_features=test_features, test_labels=test_labels)
			#random_gc_scores.append(random_gc_score)

		#least certain samples
		try:
			least_certain_active_indeces = least_confident_active_sampling(potential_active_features, impersonal_model, ts)
		except ValueError as ve:
			if "The number of personal samples provided" in ve.args[0]:
				print("Can't evaluate participant #%s with %s personal labels..."%(user_id, ts))
				continue
		sampled_active_features = potential_active_features[least_certain_active_indeces]
		sampled_active_labels = potential_active_labels[least_certain_active_indeces]

		# run personal model
		least_certain_personal_score = personal_model(sampled_active_features, sampled_active_labels, test_features=test_features, test_labels=test_labels)

		# run personal + universal
		least_certain_personal_plus_all_score = impersonal_plus_personal_model(sampled_active_features, sampled_active_labels,
																impersonal_features, impersonal_labels,
																test_features=test_features, test_labels=test_labels)

		# run personal + cluster
		least_certain_personal_plus_cluster_score = cluster_plus_personal_model(sampled_active_features, sampled_active_labels,
																impersonal_features, impersonal_labels,
																KM, clusters,
																test_features=test_features, test_labels=test_labels)


		# run garcia-ceja personalization approach
		#least_certain_gc_score = garcia_ceja_model(sampled_active_features, sampled_active_labels,
		#							impersonal_features, impersonal_labels,
		#							test_features=test_features, test_labels=test_labels)
		
		row = {"test user" : user_id,
				   "k-run" : k_run,
			   "classifier" : "RF with Wiki Parameters",
			   "personal training data" : ts,
			   "random personal score Mean" : np.mean(random_personal_scores),
			   "impersonal score Mean" : impersonal_model_score,
			   "random personal + impersonal score Mean" : np.mean(random_personal_plus_all_scores),
			   "random personal + cluster score Mean" : np.mean(random_personal_plus_cluster_scores),
			   #"random Garcia-Ceja MM Mean" : np.mean(random_gc_scores),
			   "least_certain personal score Mean" : least_certain_personal_score,
			   "least_certain personal + impersonal score Mean" : least_certain_personal_plus_all_score,
			   "least_certain personal + cluster score Mean" : least_certain_personal_plus_cluster_score,
			   #"least_certain Garcia-Ceja MM Mean" : least_certain_gc_score,
			   }
		print("\tamount of personal data : %s row" % ts)
		print("\trandom personal model score : M=%.3f, SD=%.3f" % (row["random personal score Mean"], np.std(random_personal_scores)))
		print("\timpersonal model score : M=%.3f" % (row["impersonal score Mean"]))
		print("\trandom personal + ALL Impersonal : M=%.3f, SD=%.3f" % (row["random personal + impersonal score Mean"], np.std(random_personal_plus_all_scores)))
		print("\trandom personal + CLUSTER Impersonal : M=%.3f, SD=%.3f" % (row["random personal + cluster score Mean"], np.std(random_personal_plus_cluster_scores)))
		#print("\trandom GC MM M=%.3f, SD=%.3f" % (row["random Garcia-Ceja MM Mean"], np.std(random_gc_scores)))
		print("\tleast_certain personal model score : %.3f" % (row["least_certain personal score Mean"]))
		print("\tleast_certain personal + ALL Impersonal : %.3f" % (row["least_certain personal + impersonal score Mean"]))
		print("\tleast_certain personal + CLUSTER Impersonal : %.3f" % (row["least_certain personal + cluster score Mean"]))
		#print("\tleast_certain GC MM %.3f" % (row["least_certain Garcia-Ceja MM Mean"]))
		print("\n")
		rows.append(row)

	user_scores_df = pd.DataFrame(rows)
	return user_scores_df

def pipeline1(version, output_path, user_ids, k=10, minimum_personal_samples=40, make_compatible=True):
	# initialize pipeline variables
	random_sample_iterations = 5
	
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		# Train model with v1.1 data and get clusterings
		set_data(version=version, make_compatible=make_compatible)

		for ind, user_id in enumerate(user_ids): # iterate through the users holding one out for testing
			user_results = []
			print("Running user #%s: %s" % (ind, user_id))
			personal_set = get_user_set(user_id)
			personal_set = remove_all_nan(personal_set)

			print("%s personal samples" % len(personal_set))

			if len(personal_set) < minimum_personal_samples:
				print("User %s has less than %s labeled samples..." % (user_id, minimum_personal_samples))
				continue

			personal_labels = np.array([t.decode("utf-8") for t in personal_set['class'].as_matrix()])
			personal_features = personal_set.as_matrix(columns=[personal_set.columns[1:-1]])

			# get impersonal data
			impersonal_set = data_df[data_df['user'] != user_id]
			impersonal_set = remove_all_nan(impersonal_set)
			impersonal_labels = np.array([t.decode("utf-8") for t in impersonal_set['class'].as_matrix()])
			impersonal_features = impersonal_set.as_matrix(columns=[impersonal_set.columns[1:-1]])

			# train an impersonal model
			impersonal_scaler = StandardScaler().fit(impersonal_features)
			scaled_train_x = impersonal_scaler.transform(impersonal_features)

			rfc_clf = weka_RF()
			rfc_clf.fit(scaled_train_x, impersonal_labels)

			# calibrated for probability estimation
			prob_cal_cv_generator = StratifiedKFold(n_splits=3).split(impersonal_features,impersonal_labels)
			prob_cal_clf = CalibratedClassifierCV(rfc_clf, cv=prob_cal_cv_generator, method='sigmoid')
			prob_cal_clf.fit(scaled_train_x, impersonal_labels)

			# create clusters
			number_of_clusters = 4 # the higher this number is, the smaller we should expect each cluster to be

			KM = KMeans(n_clusters=number_of_clusters)
			clusters = KM.fit_predict(scaled_train_x)

			# split personal data into training (potentially) and test
			skf = StratifiedKFold(n_splits=k)
			k_run = 0

			for active_index, test_index in skf.split(personal_features, personal_labels):
				print("\tRunning Fold #%s\n" % k_run)
				# data set available for active labeling from the individual
				all_active_features = personal_features[active_index]
				all_active_labels = personal_labels[active_index]

				# held out test set from individual
				test_features = personal_features[test_index]
				test_labels = personal_labels[test_index]
			
				k_run_df = sample_experiments(user_id, k_run,
							  impersonal_features, impersonal_labels, \
							  all_active_features, all_active_labels, \
							  test_features, test_labels, \
							  training_sizes, \
							  random_sample_iterations, \
							  impersonal_model=prob_cal_clf, impersonal_scaler=impersonal_scaler,
							  KM=KM, clusters=clusters)
				user_results.append(k_run_df)
				k_run += 1
			user_scores_df = pd.concat(user_results)
			user_scores_df.to_pickle(output_path+user_id+".pickle")

def pipeline2(output_path, user_ids, k=10, minimum_personal_samples=40):
	# initialize pipeline variables
	random_sample_iterations = 5
	
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		# Train model with v1.1 data and get clusterings
		set_data(version='1', make_compatible=True)

		data_df_v1 = remove_all_nan(data_df)
		user_ids_v1 = user_ids

		impersonal_labels = np.array([t.decode("utf-8") for t in data_df_v1['class'].as_matrix()])
		impersonal_features = data_df_v1.as_matrix(columns=[data_df_v1.columns[1:-1]])

		# train an impersonal model
		impersonal_scaler = StandardScaler().fit(impersonal_features)
		scaled_train_x = impersonal_scaler.transform(impersonal_features)

		rfc_clf = weka_RF()
		rfc_clf.fit(scaled_train_x, impersonal_labels)

		# calibrated for probability estimation
		prob_cal_clf = CalibratedClassifierCV(rfc_clf, method='sigmoid')
		prob_cal_clf.fit(scaled_train_x, impersonal_labels)

		# create clusters
		number_of_clusters = 4 # the higher this number is, the smaller we should expect each cluster to be

		KM = KMeans(n_clusters=number_of_clusters)
		clusters = KM.fit_predict(scaled_train_x)

		# reset data back to v2.0
		set_data(version="2", make_compatible=True)

		for ind, user_id in enumerate(user_ids): # iterate through the users holding one out for testing
			user_results = []
			print("Running user #%s: %s" % (ind, user_id))
			personal_set = get_user_set(user_id)
			personal_set = remove_all_nan(personal_set)

			print("%s personal samples" % len(personal_set))

			if len(personal_set) < minimum_personal_samples:
				print("User %s has less than %s labeled samples..." % (user_id, minimum_personal_samples))
				continue

			personal_labels = np.array([t.decode("utf-8") for t in personal_set['class'].as_matrix()])
			personal_features = personal_set.as_matrix(columns=[personal_set.columns[1:-1]])

			# split personal data into training (potentially) and test
			skf = StratifiedKFold(n_splits=k)
			k_run = 0

			for active_index, test_index in skf.split(personal_features, personal_labels):
				print("\tRunning Fold #%s\n" % k_run)
				# data set available for active labeling from the individual
				all_active_features = personal_features[active_index]
				all_active_labels = personal_labels[active_index]

				# held out test set from individual
				test_features = personal_features[test_index]
				test_labels = personal_labels[test_index]
			
				k_run_df = sample_experiments(user_id, k_run, 
							  impersonal_features, impersonal_labels, \
							  all_active_features, all_active_labels, \
							  test_features, test_labels, \
							  training_sizes, \
							  random_sample_iterations, \
							  impersonal_model=prob_cal_clf, impersonal_scaler=impersonal_scaler,
							  KM=KM, clusters=clusters)
				user_results.append(k_run_df)
				k_run += 1
			user_scores_df = pd.concat(user_results)
			user_scores_df.to_pickle(output_path+user_id+".pickle")