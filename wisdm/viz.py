from wisdm import wisdm
import pandas as pd
import numpy as np
from plotly import graph_objs as go
def set_data(version="2", make_compatible=True):
	wisdm.set_data(version=version, make_compatible=make_compatible)

set_data()

bad_user_ids = None
good_user_ids = None
results_df = None

def get_results_dataframe(experiment_path):
	results = []
	global bad_user_ids
	global good_user_ids
	global results_df 
	bad_user_ids = []
	good_user_ids = []
	if experiment_path[-1] is not '/':
		experiment_path += '/'
	print("Loading results from %s" % experiment_path)
	for user_id in wisdm.user_ids:
		try:
			user_results_path = experiment_path+user_id+".pickle"
			#print(user_results_path)
			user_results_df = pd.read_pickle(experiment_path+user_id+".pickle")
			results.append(user_results_df)
			good_user_ids.append(user_id)
		except FileNotFoundError as fnfe:
			#print("%s not found : user may not have had enough labeled data" % user_id)
			bad_user_ids.append(user_id)
			pass

	print("%s users loaded, %s users excluded" % (len(good_user_ids), len(bad_user_ids)))
	print("These users may not have had enough data:")
	print(bad_user_ids)			
	results_df = pd.concat(results).reset_index(drop=True)
	return results_df

def accuracy_means_by_training_size():
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	personal_means = {}
	personal_plus_universal_means = {}
	personal_plus_cluster_means = {}
	universal_means = results_df['impersonal score Mean']

	for ts in training_sizes:
		personal_scores = results_df[results_df['personal training data'] == ts]['personal score Mean']
		personal_means[ts] = personal_scores
		
		personal_plus_universal_scores = results_df[results_df['personal training data'] == ts]['personal + impersonal score Mean']
		personal_plus_universal_means[ts] = personal_plus_universal_scores
		
		personal_plus_cluster_scores = results_df[results_df['personal training data'] == ts]['personal + cluster score Mean']
		personal_plus_cluster_means[ts] = personal_plus_cluster_scores

		universal_trace = go.Box(y=universal_means,
					  x=[0]*len(universal_means),
					  name="universal model",
					  )

	data = [universal_trace]

	personal_x = []
	personal_plus_universal_x = []
	personal_plus_cluster_x = []

	personal_means_all = []
	personal_plus_universal_means_all = []
	personal_plus_cluster_means_all = []

	for ts in training_sizes:
		personal_means_all += personal_means[ts].tolist()
		personal_x += [ts] * len(personal_means[ts])
		
		personal_plus_universal_means_all += personal_plus_universal_means[ts].tolist()
		personal_plus_universal_x += [ts] *len(personal_plus_universal_means[ts])
		
		personal_plus_cluster_means_all += personal_plus_cluster_means[ts].tolist()
		personal_plus_cluster_x += [ts] * len(personal_plus_cluster_means[ts])

	personal_trace = go.Box(y=personal_means_all,
						 x=personal_x,
						 #boxpoints='all',
						 #jitter=0.8,
						 marker=dict(opacity=0.3),
						 name="Personal")

	data.append(personal_trace)

	personal_plus_universal_trace = go.Box(y=personal_plus_universal_means_all,
										x=personal_plus_universal_x,
										#boxpoints='all',
										#jitter=0.8,
										marker=dict(opacity=0.3),
										name="Personal + Universal")

	data.append(personal_plus_universal_trace)

	personal_plus_cluster_trace = go.Box(y=personal_plus_cluster_means_all,
										x=personal_plus_cluster_x,
										#boxpoints='all',
									  #jitter=0.8,
									  marker=dict(opacity=0.3),
										name="Personal + Cluster")

	data.append(personal_plus_cluster_trace)

	layout = go.Layout(showlegend=True,
					boxmode='group',
					yaxis=dict(title="Accuracy in %"),
					xaxis=dict(title="Amount of training data or training burden to the user"))
	fig = go.Figure(data=data, layout=layout)
	return fig

def plotScoresByUser(user_id):
	training_sizes = [10,20,30,40,50,60,70,80,90,100]
	personal_means = {}
	personal_plus_universal_means = {}
	personal_plus_cluster_means = {}
	universal_means = results_df['impersonal score Mean']

	for ts in training_sizes:
		personal_scores = results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)]['personal score Mean']
		personal_means[ts] = personal_scores

		personal_plus_universal_scores = results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)]['personal + impersonal score Mean']
		personal_plus_universal_means[ts] = personal_plus_universal_scores

		personal_plus_cluster_scores = results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)]['personal + cluster score Mean']
		personal_plus_cluster_means[ts] = personal_plus_cluster_scores
	
	
	universal_trace = go.Box(y=universal_means,
					  x=[0]*len(universal_means),
					  name="universal model",
					  boxpoints='suspectedoutliers',
					  )

	data = [universal_trace]

	personal_x = []
	personal_plus_universal_x = []
	personal_plus_cluster_x = []

	personal_means_all = []
	personal_plus_universal_means_all = []
	personal_plus_cluster_means_all = []

	for ts in training_sizes:
		personal_means_all += personal_means[ts].tolist()
		personal_x += [ts] * len(personal_means[ts])

		personal_plus_universal_means_all += personal_plus_universal_means[ts].tolist()
		personal_plus_universal_x += [ts] *len(personal_plus_universal_means[ts])

		personal_plus_cluster_means_all += personal_plus_cluster_means[ts].tolist()
		personal_plus_cluster_x += [ts] * len(personal_plus_cluster_means[ts])

	personal_trace = go.Box(y=personal_means_all,
						 x=personal_x,
						 name="Personal",
						 boxpoints="suspectedoutliers")

	data.append(personal_trace)

	personal_plus_universal_trace = go.Box(y=personal_plus_universal_means_all,
										x=personal_plus_universal_x,
										name="Personal + Universal",
										boxpoints="suspectedoutliers")

	data.append(personal_plus_universal_trace)

	personal_plus_cluster_trace = go.Box(y=personal_plus_cluster_means_all,
										x=personal_plus_cluster_x,
										name="Personal + Cluster",
										boxpoints="suspectedoutliers")

	data.append(personal_plus_cluster_trace)

	layout = go.Layout(showlegend=True, boxmode='group')
	fig = go.Figure(data=data, layout=layout)
	return fig

def getModelAccuracyMean(user_id, ts):
	# each list element at index, i, represents that model improvement over the best other model with training size[k]
	user_df = results_df[(results_df['test user'] == user_id) & \
						 (results_df['personal training data'] == ts)]
	personal_score_mean = user_df['personal score Mean'].mean()
	impersonal_score_mean = user_df['impersonal score Mean'].mean()
	personal_plus_impersonal_mean = user_df['personal + impersonal score Mean'].mean()
	personal_plus_cluster_mean = user_df['personal + cluster score Mean'].mean()

	#print("personal : %s" % personal_score_mean)
	#print("impersonal : %s" % impersonal_score_mean)
	#print("personal + impersonal : %s" % personal_plus_impersonal_mean)
	#print("personal + cluster : %s" % personal_plus_cluster_mean)
	mean_scores = {"personal" : personal_score_mean,
				   "impersonal" : impersonal_score_mean,
				   "personal + impersonal" : personal_plus_impersonal_mean,
				   "personal + cluster" : personal_plus_cluster_mean}
	return mean_scores

def getBests(training_size):
	model_means_columns = ['user id', 'personal', 'impersonal', 'personal + impersonal', 'personal + cluster']
	model_means = []

	for user_id in wisdm.user_ids:
		if user_id not in bad_user_ids:
			mean_scores = getModelAccuracyMean(user_id, training_size)
			mean_scores['user id'] = user_id
			model_means.append(mean_scores)

	scores_df = pd.DataFrame(model_means, columns=model_means_columns)
	
	users_benefit_from_personal = []
	users_benefit_from_impersonal = []
	users_benefit_from_personal_plus_impersonal = []
	users_benefit_from_personal_plus_cluster = []

	for ind, row in scores_df.iterrows():
		scores = [row['personal'], row['impersonal'], row['personal + impersonal'], row['personal + cluster']]
		best_model = np.argmax(scores)

		if best_model == 0:
			users_benefit_from_personal.append(row['user id'])
		elif best_model == 1:
			users_benefit_from_impersonal.append(row['user id'])
		elif best_model == 2:
			users_benefit_from_personal_plus_impersonal.append(row['user id'])
		elif best_model == 3:
			users_benefit_from_personal_plus_cluster.append(row['user id'])
	return users_benefit_from_personal, users_benefit_from_impersonal, \
			users_benefit_from_personal_plus_impersonal, users_benefit_from_personal_plus_cluster

def plotUserBests():
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	personal_bests = []
	impersonal_bests = []
	personal_impersonal_bests = []
	personal_cluster_bests = []

	for ts in training_sizes:
		personal, impersonal, personal_impersonal, personal_cluster = getBests(ts)
		
		personal_bests.append(personal)
		impersonal_bests.append(impersonal)
		personal_impersonal_bests.append(personal_impersonal)
		personal_cluster_bests.append(personal_cluster)
		
		#print("Training Size : %s" % ts)
		#print("\t personal : %s" % len(personal))
		#print("\t impersonal : %s" % len(impersonal))
		#print("\t personal + impersonal : %s" % len(personal_impersonal))
		#print("\t personal + cluster : %s" % len(personal_cluster))

	personal_trace = go.Scatter(x=training_sizes,
						 y=[len(x) for x in personal_bests],
						 name="Personal")
	impersonal_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in impersonal_bests],
						   name="Impersonal")
	personal_impersonal_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in personal_impersonal_bests],
						   name="Personal + Impersonal")
	personal_cluster_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in personal_cluster_bests],
						   name="Personal + Cluster")

	data = [personal_trace, impersonal_trace, personal_impersonal_trace, personal_cluster_trace]
	layout=go.Layout(title="WISDM V2 w/v1.1 training : #users who get the best performance from a model as the personal training set increases",
				  yaxis=dict(range=[0,40],
					  title="Number of users who get the best performance from this model"),
				  xaxis=dict(title="Amount of personal training data or user training burden"))
	fig = go.Figure(data=data, layout=layout)
	return fig
