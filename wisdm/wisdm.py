import pandas as pd
from scipy.io.arff import loadarff
from scipy.stats import randint as sp_randint
import scipy
import numpy as np
from sklearn.ensemble import RandomForestClassifier

dataset_path = "./datasets/"
wisdm_v1_dataset_path = dataset_path + "WISDM_v1/"
wisdm_v2_dataset_path = dataset_path + "WISDM_v2/"

wisdm_transformed_v1 = "WISDM_ar_v1.1_transformed_FIXED.arff"
wisdm_transformed_v2 = "WISDM_at_v2.0_transformed_FIXED.arff"

WISDM_DIR = wisdm_v1_dataset_path # switch this at anytime to fix
WISDM_TRANSFORMED = wisdm_transformed_v1

data_df = None
user_ids = None

def make_labels_compatible(data_df, version):
    if version is '2':
        # remove rows with "LyingDown" as class
        new_data_df = data_df[data_df['class'] != b"LyingDown"]
        return new_data_df
    elif (version is '1') or (version is '1.1'):
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
		data_df = make_labels_compatible

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
		clf = RandomForestClassifier(n_estimators=100, n_jobs=6, max_features="log2",
																 min_samples_leaf=1)
		return clf


