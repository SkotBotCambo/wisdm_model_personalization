import pandas as pd
from scipy.io.arff import loadarff
from scipy.stats import randint as sp_randint
import scipy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
	random_indeces = np.random.choice(len(user_ids), n)
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

	rfc_clf = wisdm.weka_RF()
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
	
	rfc_clf = wisdm.weka_RF()
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

	rfc_clf = wisdm.weka_RF()
	rfc_clf.fit(scaled_train_x, personal_plus_impersonal_labels)

	if test_features is None:
		return rfc_clf
	scaled_test_x = scaler.transform(test_features)

	
	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)
	return score

def cluster_plus_personal_model(personal_features, personal_labels,
								  impersonal_features, impersonal_labels,
								  test_features=None, test_labels=None, KM, clusters):
	cluster_predictions = KM.predict(personal_features)
	closest_cluster = mode(cluster_predictions).mode[0]

	cluster_data_indeces = [i for i in range(len(clusters)) if clusters[i] == closest_cluster]
	cluster_features = impersonal_features[cluster_data_indeces]
	cluster_labels = impersonal_labels[cluster_data_indeces]

	training_features = np.vstack((personal_features, cluster_features))
	training_labels = np.hstack((personal_labels, cluster_labels))

	scaler = StandardScaler().fit(training_features)
	scaled_train_x = scaler.transform(training_features)


	rfc_clf = wisdm.weka_RF()
	rfc_clf.fit(scaled_train_x, training_labels)

	if test_features=None:
		return rfc_clf
   
	scaled_test_x = scaler.transform(test_features)

	predictions = rfc_clf.predict(scaled_test_x)
	score = accuracy_score(test_labels, predictions)
	return score

def random_sample_experiments(personal_features, personal_labels, \
							  impersonal_features, impersonal_labels, \
							  active_features, active_labels, \
							  training_sizes, \
							  random_iterations, \
							  impersonal_model=None, KM=None, \
							  clusters=None)
	# iteratively increase the number of samples we use
	user_scores_df = []
	for ts in training_sizes:
		# initialize score holders
		personal_model_scores = []
		impersonal_model_scores = []
		personal_plus_all_scores = []
		personal_plus_cluster_scores = []

		# run impersonal model
		impersonal_scaled_test_x = impersonal_scaler.transform(test_features)
		impersonal_model_score = accuracy_score(test_labels, rfc_clf.predict(impersonal_scaled_test_x))
		impersonal_model_scores.append(universal_model_score)

		for run in range(random_sample_iterations):
			# get random samples
			active_indeces = np.random.choice(len(personal_set), ts)
			sampled_active_features = personal_features[active_indeces]
			sampled_active_labels = personal_labels[active_indeces]

			# run personal model
			personal_score = personal_model(sampled_active_features, sampled_active_labels, test_features=test_features, test_labels=test_labels)
			personal_model_scores.append(personal_score)

			# run personal + universal
			personal_plus_all_score = impersonal_plus_personal_model(sampled_active_features, sampled_active_labels,
																	impersonal_features, impersonal_labels,
																	test_features=test_features, test_labels=test_labels)
			personal_plus_all_scores.append(personal_plus_all_score)

			# run personal + cluster
			personal_plus_cluster_score = cluster_plus_personal_model(sampled_active_features, sampled_active_labels,
																	impersonal_features, impersonal_labels,
																	test_features=test_features, test_labels=test_labels, KM, clusters)

			personal_plus_cluster_scores.append(personal_plus_cluster_score)
		row = {"test user" : user_id,
                   "k-run" : k_run,
               "classifier" : "RF with Wiki Parameters",
               "personal training data" : training_size,
               "personal score Mean" : np.mean(personal_model_scores),
               "personal score STD" : np.std(personal_model_scores),
               "impersonal score Mean" : np.mean(impersonal_model_scores),
               "impersonal score STD" : np.std(impersonal_model_scores),
               "personal + impersonal score Mean" : np.mean(personal_plus_all_scores),
               "personal + impersonal score STD" : np.std(personal_plus_all_scores),
               "personal + cluster score Mean" : np.mean(personal_plus_cluster_scores),
               "personal + cluster score STD" : np.std(personal_plus_cluster_scores)
               }
		print("\tamount of personal data : %s row" % training_size)
		print("\tpersonal model score : M=%.3f, SD=%.3f" % (row["personal score Mean"], row["personal score STD"]))
		print("\tuniversal model score : M=%.3f, SD=%.3f" % (row["impersonal score Mean"], row["impersonal score STD"]))
		print("\tpersonal + ALL Impersonal : M=%.3f, SD=%.3f" % (row["personal + impersonal score Mean"], row["personal + impersonal score STD"]))
		print("\tpersonal + CLUSTER Impersonal : M=%.3f, SD=%.3f" % (row["personal + cluster score Mean"], row["personal + cluster score STD"]))
		print("\n")
	return user_scores_df

def pipeline1(output_path, user_ids, minimum_personal_samples=40):
	# initialize pipeline variables
	random_sample_iterations = 5
	
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		# Train model with v1.1 data and get clusterings
		wisdm.set_data(version='1', make_compatible=True)

		data_df_v1 = wisdm.remove_all_nan(wisdm.data_df)
		user_ids_v1 = wisdm.user_ids

		impersonal_labels = np.array([t.decode("utf-8") for t in data_df_v1['class'].as_matrix()])
		impersonal_features = data_df_v1.as_matrix(columns=[data_df_v1.columns[1:-1]])

		# train an impersonal model
		impersonal_scaler = StandardScaler().fit(impersonal_features)
		scaled_train_x = impersonal_scaler.transform(impersonal_features)

		rfc_clf = wisdm.weka_RF()
		rfc_clf.fit(scaled_train_x, impersonal_labels)

		# calibrated for probability estimation
		prob_cal_clf = CalibratedClassifierCV(rfc_clf, method='sigmoid')
		prob_cal_clf.fit(scaled_train_x, impersonal_labels)

		for ind, user_id in enumerate(user_ids): # iterate through the users holding one out for testing
			user_scores_df = []
			print("Running user #%s: %s" % (ind, user_id))
			personal_set = wisdm.get_user_set(user_id)
			personal_set = wisdm.remove_all_nan(personal_set)

			print("%s personal samples" % len(personal_set))

			if len(personal_set) < minimum_personal_samples:
				print("User %s has less than %s labeled samples..." % (user_id, minimum_personal_samples))
				continue

			personal_labels = np.array([t.decode("utf-8") for t in personal_set['class'].as_matrix()])
			personal_features = personal_set.as_matrix(columns=[personal_set.columns[1:-1]])

			# split personal data into training (potentially) and test
			skf = StratifiedKFold(n_splits=k)
			k_run = 0

			# get test set
			# held out test set from individual
			test_features = personal_features[test_index]
			test_labels = personal_labels[test_index]

			for active_index, test_index in skf.split(personal_features, personal_labels):
				#print("\tRunning Fold #%s\n" % k_run)
				# data set available for active labeling from the individual
				all_active_features = personal_features[active_index]
				all_active_labels = personal_labels[active_index]
			
				# iteratively increase the number of samples we use
				for ts in training_sizes:
					# initialize score holders
					personal_model_scores = []
					impersonal_model_scores = []
					personal_plus_all_scores = []
					personal_plus_cluster_scores = []

					# run impersonal model
					impersonal_scaled_test_x = impersonal_scaler.transform(test_features)
					impersonal_model_score = accuracy_score(test_labels, rfc_clf.predict(impersonal_scaled_test_x))
					impersonal_model_scores.append(universal_model_score)

					for run in range(random_sample_iterations):
						# get random samples
						active_indeces = np.random.choice(len(personal_set), ts)
						sampled_active_features = personal_features[active_indeces]
						sampled_active_labels = personal_labels[active_indeces]

						# run personal model
						personal_score = personal_model(sampled_active_features, sampled_active_labels, test_features=test_features, test_labels=test_labels)
						personal_model_scores.append(personal_score)

						# run personal + universal
						personal_plus_all_score = impersonal_plus_personal_model(sampled_active_features, sampled_active_labels,
																				impersonal_features, impersonal_labels,
																				test_features=test_features, test_labels=test_labels)
						personal_plus_all_scores.append(personal_plus_all_score)

						# run personal + cluster
						personal_plus_cluster_score = cluster_plus_personal_model(sampled_active_features, sampled_active_labels,
																				impersonal_features, impersonal_labels,
																				test_features=test_features, test_labels=test_labels, KM, clusters)

						personal_plus_cluster_scores.append(personal_plus_cluster_score)
					row = {"test user" : user_id,
                               "k-run" : k_run,
                           "classifier" : "RF with Wiki Parameters",
                           "personal training data" : training_size,
                           "personal score Mean" : np.mean(personal_model_scores),
                           "personal score STD" : np.std(personal_model_scores),
                           "impersonal score Mean" : np.mean(impersonal_model_scores),
                           "impersonal score STD" : np.std(impersonal_model_scores),
                           "personal + impersonal score Mean" : np.mean(personal_plus_all_scores),
                           "personal + impersonal score STD" : np.std(personal_plus_all_scores),
                           "personal + cluster score Mean" : np.mean(personal_plus_cluster_scores),
                           "personal + cluster score STD" : np.std(personal_plus_cluster_scores)
                           }
					print("\tamount of personal data : %s row" % training_size)
					print("\tpersonal model score : M=%.3f, SD=%.3f" % (row["personal score Mean"], row["personal score STD"]))
					print("\tuniversal model score : M=%.3f, SD=%.3f" % (row["impersonal score Mean"], row["impersonal score STD"]))
					print("\tpersonal + ALL Impersonal : M=%.3f, SD=%.3f" % (row["personal + impersonal score Mean"], row["personal + impersonal score STD"]))
					print("\tpersonal + CLUSTER Impersonal : M=%.3f, SD=%.3f" % (row["personal + cluster score Mean"], row["personal + cluster score STD"]))
					print("\n")
					user_scores_df.append(row)
				k_run += 1