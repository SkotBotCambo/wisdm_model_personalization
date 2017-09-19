import ipyparallel as ipp
import sys

c = ipp.Client()
dview = c[:]
dview.block=True

print("Setting up parallelization...")
with dview.sync_imports():
	import sys

results = dview.execute('sys.path.append("/home/sac086/wisdm_model_personalization/")')

print("Importing modules...\n")
with dview.sync_imports():
	import warnings
	import os
	from wisdm import wisdm
	import numpy as np
	import pandas as pd
	from sklearn.metrics import accuracy_score
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
	from sklearn.cluster import KMeans
	from scipy.stats import mode
	from collections import Counter
	import time

print("\n")

description1 = '''
Training set = WISDM v1.1
Test Set = WISDM v1.1
validation = leave-one user out, class label stratified 10-fold cross-validation within user where the training folds
are used as the pool from which we can actively sample
Sampling Methods = Least-Certain Sampling, Random Sampling
Modeling Methods = Impersonal, Sampled Personal, Sampled Personal data + All Impersonal Data, 
                    Sampled Personal Data + nearest cluster of Impersonal data,
                    Sampled Personal Data + Impersonal Data selected and weighted with Garcia-Ceja approach
'''
experiment_name1 = "experiment_09-05_train_v1_with_random_and_least_certain/"
experiment_output_path1 = "/home/sac086/wisdm_model_personalization/results/" + experiment_name1
command1 = """wisdm.pipeline1('1', '%s', user_ids, minimum_personal_samples=110)""" % experiment_output_path1


if not os.path.exists(experiment_output_path1):
    os.makedirs(experiment_output_path1)
with open(experiment_output_path1 + "README.md", "w") as fOut:
    fOut.write(description1)

# divide the user ids up among the different 
wisdm.set_data(version="1", make_compatible=True)
unfinished_user_ids = ['17', '16', '28', '4', '30', '21', '9', '25', '2']
scatter_result = dview.scatter("user_ids", unfinished_user_ids)

dview.execute('''wisdm.set_data(version="1", make_compatible=True)''')

#print("\nstarting 1st experiment")
#start = time.time()
#try:
#	results = dview.execute(command1)
#except ipp.CompositeError as e:
#	e.raise_exception()
#finish = time.time()
#print("Finished all models in %s minutes" % ((finish - start) / 60.))

description2 = '''
Training set = WISDM v2.0
Test Set = WISDM v2.0
validation = leave-one user out, class label stratified 10-fold cross-validation within user where the training folds
are used as the pool from which we can actively sample
Sampling Methods = Least-Certain Sampling, Random Sampling
Modeling Methods = Impersonal, Sampled Personal, Sampled Personal data + All Impersonal Data, 
                    Sampled Personal Data + nearest cluster of Impersonal data,
                    Sampled Personal Data + Impersonal Data selected and weighted with Garcia-Ceja approach
'''

experiment_name2 = "experiment_09-05_train_v2_with_random_and_least_certain/"
experiment_output_path2 = "/home/sac086/wisdm_model_personalization/results/" + experiment_name2
command2 = """wisdm.pipeline1('2', '%s', user_ids, minimum_personal_samples=110)""" % experiment_output_path2

if not os.path.exists(experiment_output_path2):
    os.makedirs(experiment_output_path2)
with open(experiment_output_path2 + "README.md", "w") as fOut:
    fOut.write(description2)

dview.execute('''wisdm.set_data(version="2", make_compatible=True)''')

# divide the user ids up among the different 
wisdm.set_data(version="2", make_compatible=True)
scatter_result = dview.scatter("user_ids", wisdm.user_ids)

#print("\nstarting 2nd experiment")
#start = time.time()
#try:
#	results = dview.execute(command2)
#except ipp.CompositeError as e:
#	e.raise_exception()
#finish = time.time()
#print("Finished all models in %s minutes" % ((finish - start) / 60.))

description3 = '''
Training set = WISDM v1.1
Test Set = WISDM v2.0
validation = leave-one user out, class label stratified 10-fold cross-validation within user where the training folds
are used as the pool from which we can actively sample
Sampling Methods = Least-Certain Sampling, Random Sampling
Modeling Methods = Impersonal, Sampled Personal, Sampled Personal data + All Impersonal Data, 
                    Sampled Personal Data + nearest cluster of Impersonal data,
                    Sampled Personal Data + Impersonal Data selected and weighted with Garcia-Ceja approach
'''

experiment_name3 = "experiment_09-05_train_v1_test_v2_with_random_and_least_certain/"
experiment_output_path3 = "/home/sac086/wisdm_model_personalization/results/" + experiment_name3
command3 = """wisdm.pipeline2('%s', user_ids, minimum_personal_samples=110)""" % experiment_output_path3

if not os.path.exists(experiment_output_path3):
    os.makedirs(experiment_output_path3)
with open(experiment_output_path3 + "README.md", "w") as fOut:
    fOut.write(description3)

dview.execute('''wisdm.set_data(version="2", make_compatible=True)''')
# divide the user ids up among the different 
wisdm.set_data(version="2", make_compatible=True)
scatter_result = dview.scatter("user_ids", wisdm.user_ids)
print("\nstarting 3rd experiment")

start = time.time()
try:
	results = dview.execute(command3)
except ipp.CompositeError as e:
	e.raise_exception()
finish = time.time()
print("Finished all models in %s minutes" % ((finish - start) / 60.))