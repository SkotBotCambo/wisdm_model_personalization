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

def accuracy_by_training_size():
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	random_personal_means = {}
	random_personal_plus_universal_means = {}
	random_personal_plus_cluster_means = {}
	least_certain_personal_means = {}
	least_certain_personal_plus_universal_means = {}
	least_certain_personal_plus_cluster_means = {}
	universal_means = results_df['impersonal score Mean']

	for ts in training_sizes:
		random_personal_scores = results_df[results_df['personal training data'] == ts]['random personal score Mean']
		random_personal_means[ts] = random_personal_scores
		
		random_personal_plus_universal_scores = results_df[results_df['personal training data'] == ts]['random personal + impersonal score Mean']
		random_personal_plus_universal_means[ts] = random_personal_plus_universal_scores
		
		random_personal_plus_cluster_scores = results_df[results_df['personal training data'] == ts]['random personal + cluster score Mean']
		random_personal_plus_cluster_means[ts] = random_personal_plus_cluster_scores

		least_certain_personal_scores = results_df[results_df['personal training data'] == ts]['least_certain personal score Mean']
		least_certain_personal_means[ts] = least_certain_personal_scores
		
		least_certain_personal_plus_universal_scores = results_df[results_df['personal training data'] == ts]['least_certain personal + impersonal score Mean']
		least_certain_personal_plus_universal_means[ts] = least_certain_personal_plus_universal_scores
		
		least_certain_personal_plus_cluster_scores = results_df[results_df['personal training data'] == ts]['least_certain personal + cluster score Mean']
		least_certain_personal_plus_cluster_means[ts] = least_certain_personal_plus_cluster_scores

	universal_trace = go.Box(y=universal_means,
				  x=[0]*len(universal_means),
				  name="universal model",
				  )

	data = [universal_trace]

	random_personal_x = []
	random_personal_plus_universal_x = []
	random_personal_plus_cluster_x = []

	random_personal_means_all = []
	random_personal_plus_universal_means_all = []
	random_personal_plus_cluster_means_all = []

	least_certain_personal_x = []
	least_certain_personal_plus_universal_x = []
	least_certain_personal_plus_cluster_x = []

	least_certain_personal_means_all = []
	least_certain_personal_plus_universal_means_all = []
	least_certain_personal_plus_cluster_means_all = []

	for ts in training_sizes:
		random_personal_means_all += random_personal_means[ts].tolist()
		random_personal_x += [ts] * len(random_personal_means[ts])
		
		random_personal_plus_universal_means_all += random_personal_plus_universal_means[ts].tolist()
		random_personal_plus_universal_x += [ts] *len(random_personal_plus_universal_means[ts])
		
		random_personal_plus_cluster_means_all += random_personal_plus_cluster_means[ts].tolist()
		random_personal_plus_cluster_x += [ts] * len(random_personal_plus_cluster_means[ts])

		least_certain_personal_means_all += least_certain_personal_means[ts].tolist()
		least_certain_personal_x += [ts] * len(least_certain_personal_means[ts])
		
		least_certain_personal_plus_universal_means_all += least_certain_personal_plus_universal_means[ts].tolist()
		least_certain_personal_plus_universal_x += [ts] *len(least_certain_personal_plus_universal_means[ts])
		
		least_certain_personal_plus_cluster_means_all += least_certain_personal_plus_cluster_means[ts].tolist()
		least_certain_personal_plus_cluster_x += [ts] * len(least_certain_personal_plus_cluster_means[ts])

	random_personal_trace = go.Box(y=random_personal_means_all,
						 x=random_personal_x,
						 #boxpoints='all',
						 #jitter=0.8,
						 marker=dict(opacity=0.3),
						 name="Random Personal")

	data.append(random_personal_trace)

	random_personal_plus_universal_trace = go.Box(y=random_personal_plus_universal_means_all,
										x=random_personal_plus_universal_x,
										#boxpoints='all',
										#jitter=0.8,
										marker=dict(opacity=0.3),
										name="Random Personal + Universal")

	data.append(random_personal_plus_universal_trace)

	random_personal_plus_cluster_trace = go.Box(y=random_personal_plus_cluster_means_all,
										x=random_personal_plus_cluster_x,
										#boxpoints='all',
									  #jitter=0.8,
									  marker=dict(opacity=0.3),
										name="Ranomd Personal + Cluster")

	data.append(random_personal_plus_cluster_trace)

	least_certain_personal_trace = go.Box(y=least_certain_personal_means_all,
						 x=least_certain_personal_x,
						 #boxpoints='all',
						 #jitter=0.8,
						 marker=dict(opacity=0.3),
						 name="Least Certain Personal")

	data.append(least_certain_personal_trace)

	least_certain_personal_plus_universal_trace = go.Box(y=least_certain_personal_plus_universal_means_all,
										x=least_certain_personal_plus_universal_x,
										#boxpoints='all',
										#jitter=0.8,
										marker=dict(opacity=0.3),
										name="Least Certain Personal + Universal")

	data.append(least_certain_personal_plus_universal_trace)

	least_certain_personal_plus_cluster_trace = go.Box(y=least_certain_personal_plus_cluster_means_all,
										x=least_certain_personal_plus_cluster_x,
										#boxpoints='all',
									  #jitter=0.8,
									  marker=dict(opacity=0.3),
										name="Least Certain Personal + Cluster")

	data.append(least_certain_personal_plus_cluster_trace)

	layout = go.Layout(showlegend=True,
					boxmode='group',
					yaxis=dict(title="Accuracy in %"),
					xaxis=dict(title="Amount of training data or training burden to the user"))
	fig = go.Figure(data=data, layout=layout)
	return fig

def plotScoresByUser(user_id):
	training_sizes = [10,20,30,40,50,60,70,80,90,100]
	random_personal_means = {}
	random_personal_plus_universal_means = {}
	random_personal_plus_cluster_means = {}
	least_certain_personal_means = {}
	least_certain_personal_plus_universal_means = {}
	least_certain_personal_plus_cluster_means = {}
	universal_means = results_df['impersonal score Mean']

	for ts in training_sizes:
		random_personal_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['random personal score Mean']
		random_personal_means[ts] = random_personal_scores
		
		random_personal_plus_universal_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['random personal + impersonal score Mean']
		random_personal_plus_universal_means[ts] = random_personal_plus_universal_scores
		
		random_personal_plus_cluster_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['random personal + cluster score Mean']
		random_personal_plus_cluster_means[ts] = random_personal_plus_cluster_scores

		least_certain_personal_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['least_certain personal score Mean']
		least_certain_personal_means[ts] = least_certain_personal_scores
		
		least_certain_personal_plus_universal_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['least_certain personal + impersonal score Mean']
		least_certain_personal_plus_universal_means[ts] = least_certain_personal_plus_universal_scores
		
		least_certain_personal_plus_cluster_scores = results_df[results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)] == ts]['least_certain personal + cluster score Mean']
		least_certain_personal_plus_cluster_means[ts] = least_certain_personal_plus_cluster_scores
		personal_scores = results_df[(results_df['personal training data'] == ts) & \
									 (results_df['test user'] == user_id)]['personal score Mean']
	
	
	universal_trace = go.Box(y=universal_means,
					  x=[0]*len(universal_means),
					  name="universal model",
					  boxpoints='suspectedoutliers',
					  )

	data = [universal_trace]

	random_personal_x = []
	random_personal_plus_universal_x = []
	random_personal_plus_cluster_x = []

	random_personal_means_all = []
	random_personal_plus_universal_means_all = []
	random_personal_plus_cluster_means_all = []

	least_certain_personal_x = []
	least_certain_personal_plus_universal_x = []
	least_certain_personal_plus_cluster_x = []

	least_certain_personal_means_all = []
	least_certain_personal_plus_universal_means_all = []
	least_certain_personal_plus_cluster_means_all = []

	for ts in training_sizes:
		random_personal_means_all += random_personal_means[ts].tolist()
		random_personal_x += [ts] * len(random_personal_means[ts])
		
		random_personal_plus_universal_means_all += random_personal_plus_universal_means[ts].tolist()
		random_personal_plus_universal_x += [ts] *len(random_personal_plus_universal_means[ts])
		
		random_personal_plus_cluster_means_all += random_personal_plus_cluster_means[ts].tolist()
		random_personal_plus_cluster_x += [ts] * len(random_personal_plus_cluster_means[ts])

		least_certain_personal_means_all += least_certain_personal_means[ts].tolist()
		least_certain_personal_x += [ts] * len(least_certain_personal_means[ts])
		
		least_certain_personal_plus_universal_means_all += least_certain_personal_plus_universal_means[ts].tolist()
		least_certain_personal_plus_universal_x += [ts] *len(least_certain_personal_plus_universal_means[ts])
		
		least_certain_personal_plus_cluster_means_all += least_certain_personal_plus_cluster_means[ts].tolist()
		least_certain_personal_plus_cluster_x += [ts] * len(least_certain_personal_plus_cluster_means[ts])

	random_personal_trace = go.Box(y=random_personal_means_all,
						 x=random_personal_x,
						 #boxpoints='all',
						 #jitter=0.8,
						 marker=dict(opacity=0.3),
						 name="Random Personal")

	data.append(random_personal_trace)

	random_personal_plus_universal_trace = go.Box(y=random_personal_plus_universal_means_all,
										x=random_personal_plus_universal_x,
										#boxpoints='all',
										#jitter=0.8,
										marker=dict(opacity=0.3),
										name="Random Personal + Universal")

	data.append(random_personal_plus_universal_trace)

	random_personal_plus_cluster_trace = go.Box(y=random_personal_plus_cluster_means_all,
										x=random_personal_plus_cluster_x,
										#boxpoints='all',
									  #jitter=0.8,
									  marker=dict(opacity=0.3),
										name="Ranomd Personal + Cluster")

	data.append(random_personal_plus_cluster_trace)

	least_certain_personal_trace = go.Box(y=least_certain_personal_means_all,
						 x=least_certain_personal_x,
						 #boxpoints='all',
						 #jitter=0.8,
						 marker=dict(opacity=0.3),
						 name="Least Certain Personal")

	data.append(least_certain_personal_trace)

	least_certain_personal_plus_universal_trace = go.Box(y=least_certain_personal_plus_universal_means_all,
										x=least_certain_personal_plus_universal_x,
										#boxpoints='all',
										#jitter=0.8,
										marker=dict(opacity=0.3),
										name="Least Certain Personal + Universal")

	data.append(least_certain_personal_plus_universal_trace)

	least_certain_personal_plus_cluster_trace = go.Box(y=least_certain_personal_plus_cluster_means_all,
										x=least_certain_personal_plus_cluster_x,
										#boxpoints='all',
									  #jitter=0.8,
									  marker=dict(opacity=0.3),
										name="Least Certain Personal + Cluster")

	data.append(least_certain_personal_plus_cluster_trace)

	layout = go.Layout(showlegend=True,
					boxmode='group',
					yaxis=dict(title="Accuracy in %"),
					xaxis=dict(title="Amount of training data or training burden to the user"))
	fig = go.Figure(data=data, layout=layout)
	return fig

def getModelAccuracyMean(user_id, ts):
	# each list element at index, i, represents that model improvement over the best other model with training size[k]
	user_df = results_df[(results_df['test user'] == user_id) & \
						 (results_df['personal training data'] == ts)]

	impersonal_score_mean = user_df['impersonal score Mean'].mean()

	random_personal_score_mean = user_df['random personal score Mean'].mean()
	random_personal_plus_impersonal_mean = user_df['random personal + impersonal score Mean'].mean()
	random_personal_plus_cluster_mean = user_df['random personal + cluster score Mean'].mean()

	least_certain_personal_score_mean = user_df['least_certain personal score Mean'].mean()
	least_certain_personal_plus_impersonal_mean = user_df['least_certain personal + impersonal score Mean'].mean()
	least_certain_personal_plus_cluster_mean = user_df['least_certain personal + cluster score Mean'].mean()

	#print("personal : %s" % personal_score_mean)
	#print("impersonal : %s" % impersonal_score_mean)
	#print("personal + impersonal : %s" % personal_plus_impersonal_mean)
	#print("personal + cluster : %s" % personal_plus_cluster_mean)
	mean_scores = {"impersonal" : impersonal_score_mean,
				   "random personal" : random_personal_score_mean,
				   "random personal + impersonal" : random_personal_plus_impersonal_mean,
				   "random personal + cluster" : random_personal_plus_cluster_mean,
				   "least_certain personal" : least_certain_personal_score_mean,
				   "least_certain personal + impersonal" : least_certain_personal_plus_impersonal_mean,
				   "least_certain personal + cluster" : least_certain_personal_plus_cluster_mean,}
	return mean_scores

def getBests(training_size):
	model_means_columns = ['user id', \
						   'impersonal',\
						   'random personal',  \
						   'random personal + impersonal', \
						   'random personal + cluster', \
						   'least_certain personal',  \
						   'least_certain personal + impersonal', \
						   'least_certain personal + cluster']
	model_means = []

	for user_id in wisdm.user_ids:
		if user_id not in bad_user_ids:
			mean_scores = getModelAccuracyMean(user_id, training_size)
			mean_scores['user id'] = user_id
			model_means.append(mean_scores)

	scores_df = pd.DataFrame(model_means, columns=model_means_columns)
	
	users_benefit_from_impersonal = []
	users_benefit_from_random_personal = []
	users_benefit_from_random_personal_plus_impersonal = []
	users_benefit_from_random_personal_plus_cluster = []
	users_benefit_from_least_certain_personal = []
	users_benefit_from_least_certain_personal_plus_impersonal = []
	users_benefit_from_least_certain_personal_plus_cluster = []

	for ind, row in scores_df.iterrows():
		scores = [row['impersonal'], \
				  row['random personal'], \
				  row['random personal + impersonal'], \
				  row['random personal + cluster'], \
				  row['least_certain personal'], \
				  row['least_certain personal + impersonal'], \
				  row['least_certain personal + cluster'], ]
		best_model = np.argmax(scores)

		if best_model == 0:
			users_benefit_from_impersonal.append(row['user id'])
		elif best_model == 1:
			users_benefit_from_random_personal.append(row['user id'])
		elif best_model == 2:
			users_benefit_from_random_personal_plus_impersonal.append(row['user id'])
		elif best_model == 3:
			users_benefit_from_random_personal_plus_cluster.append(row['user id'])
		elif best_model == 4:
			users_benefit_from_least_certain_personal.append(row['user id'])
		elif best_model == 5:
			users_benefit_from_least_certain_personal_plus_impersonal.append(row['user id'])
		elif best_model == 6:
			users_benefit_from_least_certain_personal_plus_cluster.append(row['user id'])
	return users_benefit_from_impersonal, \
			users_benefit_from_random_personal, \
			users_benefit_from_random_personal_plus_impersonal, \
			users_benefit_from_random_personal_plus_cluster, \
			users_benefit_from_least_certain_personal, \
			users_benefit_from_least_certain_personal_plus_impersonal, \
			users_benefit_from_least_certain_personal_plus_cluster

def plotUserBestsAreaStack():
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	impersonal_bests = []
	random_personal_bests = []
	random_personal_impersonal_bests = []
	random_personal_cluster_bests = []
	least_certain_personal_bests = []
	least_certain_personal_impersonal_bests = []
	least_certain_personal_cluster_bests = []

	for ts in training_sizes:
		impersonal, random_personal, random_personal_impersonal, random_personal_cluster, \
			least_certain_personal, least_certain_personal_impersonal, least_certain_personal_cluster = getBests(ts)

		impersonal_bests.append(impersonal)		
		random_personal_bests.append(random_personal)
		random_personal_impersonal_bests.append(random_personal_impersonal)
		random_personal_cluster_bests.append(random_personal_cluster)
		least_certain_personal_bests.append(least_certain_personal)
		least_certain_personal_impersonal_bests.append(least_certain_personal_impersonal)
		least_certain_personal_cluster_bests.append(least_certain_personal_cluster)

		#print("Training Size : %s" % ts)
		#print("\t personal : %s" % len(personal))
		#print("\t impersonal : %s" % len(impersonal))
		#print("\t personal + impersonal : %s" % len(personal_impersonal))
		#print("\t personal + cluster : %s" % len(personal_cluster))   
	impersonal_stack = [len(y) for y in impersonal_bests]
	random_personal_stack = [len(y0)+len(y1) for y0,y1 in zip(impersonal_bests, random_personal_bests)]
	random_personal_impersonal_stack = [y0+len(y1) for y0,y1 in \
										zip(random_personal_stack, random_personal_impersonal_bests)]
	random_personal_cluster_stack = [y0+len(y1) for y0,y1 in \
									 zip(random_personal_impersonal_stack, random_personal_cluster_bests)]
	least_certain_personal_stack = [y0+len(y1) for y0,y1 in \
									 zip(random_personal_cluster_stack, least_certain_personal_bests)]
	least_certain_personal_impersonal_stack = [y0+len(y1) for y0,y1 in \
									 zip(least_certain_personal_stack, least_certain_personal_impersonal_bests)]
	least_certain_personal_cluster_stack = [y0+len(y1) for y0,y1 in \
									 zip(least_certain_personal_impersonal_stack, least_certain_personal_cluster_bests)]

	impersonal_txt = [str(len(y)) for y in impersonal_bests]
	random_personal_txt = [str(len(y)) for y in random_personal_bests]
	random_personal_impersonal_txt = [str(len(y)) for y in random_personal_impersonal_bests]
	random_personal_cluster_txt = [str(len(y)) for y in random_personal_cluster_bests]
	least_certain_personal_txt = [str(len(y)) for y in least_certain_personal_bests]
	least_certain_personal_impersonal_txt = [str(len(y)) for y in least_certain_personal_impersonal_bests]
	least_certain_personal_cluster_txt = [str(len(y)) for y in least_certain_personal_cluster_bests]

	impersonal_trace = go.Scatter(x=training_sizes,
						   y=impersonal_stack,
						   text=impersonal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						   name="Impersonal")
	random_personal_trace = go.Scatter(x=training_sizes,
						 y=random_personal_stack,
						   text=random_personal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						 name="Random Personal")
	random_personal_impersonal_trace = go.Scatter(x=training_sizes,
						   y=random_personal_impersonal_stack,
						   text=random_personal_impersonal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						   name="Random Personal + Impersonal")
	random_personal_cluster_trace = go.Scatter(x=training_sizes,
						   y=random_personal_cluster_stack,
						   text=impersonal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						   name="Random Personal + Cluster")

	least_certain_personal_trace = go.Scatter(x=training_sizes,
						 y=least_certain_personal_stack,
						   text=least_certain_personal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						 name="Least Certain Personal")
	least_certain_personal_impersonal_trace = go.Scatter(x=training_sizes,
						   y=least_certain_personal_impersonal_stack,
						   text=least_certain_personal_impersonal_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						   name="Least Certain Personal + Impersonal")
	least_certain_personal_cluster_trace = go.Scatter(x=training_sizes,
						   y=least_certain_personal_cluster_stack,
						   text=least_certain_personal_cluster_txt,
						   hoverinfo="x+text",
						   mode='lines',
						   line=dict(width=0.5),
						   fill='tonexty',
						   name="Least Certain Personal + Cluster")

	data = [impersonal_trace, \
			random_personal_trace, \
			random_personal_impersonal_trace, \
			random_personal_cluster_trace,
			least_certain_personal_trace, \
			least_certain_personal_impersonal_trace, \
			least_certain_personal_cluster_trace]

	layout=go.Layout(title="WISDM V2 w/v1.1 training : #users who get the best performance from a model as the personal training set increases",
				  yaxis=dict(title="Number of users who get the best performance from this model"),
				  xaxis=dict(title="Amount of personal training data or user training burden"))
	fig = go.Figure(data=data, layout=layout)
	return fig


def plotUserBests():
	training_sizes = [10,20,30,40,50,60,70,80,90,100]

	impersonal_bests = []
	random_personal_bests = []
	random_personal_impersonal_bests = []
	random_personal_cluster_bests = []
	least_certain_personal_bests = []
	least_certain_personal_impersonal_bests = []
	least_certain_personal_cluster_bests = []

	for ts in training_sizes:
		impersonal, random_personal, random_personal_impersonal, random_personal_cluster, \
			least_certain_personal, least_certain_personal_impersonal, least_certain_personal_cluster = getBests(ts)

		impersonal_bests.append(impersonal)		
		random_personal_bests.append(random_personal)
		random_personal_impersonal_bests.append(random_personal_impersonal)
		random_personal_cluster_bests.append(random_personal_cluster)
		least_certain_personal_bests.append(least_certain_personal)
		least_certain_personal_impersonal_bests.append(least_certain_personal_impersonal)
		least_certain_personal_cluster_bests.append(least_certain_personal_cluster)
		
		#print("Training Size : %s" % ts)
		#print("\t personal : %s" % len(personal))
		#print("\t impersonal : %s" % len(impersonal))
		#print("\t personal + impersonal : %s" % len(personal_impersonal))
		#print("\t personal + cluster : %s" % len(personal_cluster))

	impersonal_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in impersonal_bests],
						   name="Impersonal")
	random_personal_trace = go.Scatter(x=training_sizes,
						 y=[len(x) for x in random_personal_bests],
						 name="Random Personal")
	random_personal_impersonal_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in random_personal_impersonal_bests],
						   name="Random Personal + Impersonal")
	random_personal_cluster_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in random_personal_cluster_bests],
						   name="Random Personal + Cluster")

	least_certain_personal_trace = go.Scatter(x=training_sizes,
						 y=[len(x) for x in least_certain_personal_bests],
						 name="Least Certain Personal")
	least_certain_personal_impersonal_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in least_certain_personal_impersonal_bests],
						   name="Least Certain Personal + Impersonal")
	least_certain_personal_cluster_trace = go.Scatter(x=training_sizes,
						   y=[len(x) for x in least_certain_personal_cluster_bests],
						   name="Least Certain Personal + Cluster")

	data = [impersonal_trace, \
			random_personal_trace, \
			random_personal_impersonal_trace, \
			random_personal_cluster_trace,
			least_certain_personal_trace, \
			least_certain_personal_impersonal_trace, \
			least_certain_personal_cluster_trace]
	layout=go.Layout(title="WISDM V2 w/v1.1 training : #users who get the best performance from a model as the personal training set increases",
				  yaxis=dict(range=[0,40],
					  title="Number of users who get the best performance from this model"),
				  xaxis=dict(title="Amount of personal training data or user training burden"))
	fig = go.Figure(data=data, layout=layout)
	return fig

def get_performance_means(results_df, sampling_method='random'):
	test_user_ids = results_df['test user'].unique()
	training_sizes = results_df['personal training data'].unique()
	personal_models = ['personal', 'personal + impersonal', 'personal + cluster']
	models_dict = {}
	
	# get impersonal scores
	models_dict['impersonal'] = {}

	for user_id in test_user_ids:
		impersonal_accuracies = results_df[(results_df['test user'] == user_id) & (results_df['personal training data'] == 10)]['impersonal score Mean']

		models_dict['impersonal'][user_id] = np.mean(impersonal_accuracies)

	for model in personal_models:
		model_dict = {'training_sizes' : {}}
					  
		for ts in training_sizes:
			model_dict['training_sizes'][ts] = {}
			for user_id in test_user_ids:
				column_name = "%s %s score Mean" % (sampling_method, model)
				user_model_series = results_df[(results_df['personal training data'] == ts) & \
						   (results_df['test user'] == user_id)][column_name]
				model_dict['training_sizes'][ts][user_id] = user_model_series.mean()
		models_dict[model] = model_dict
	return models_dict

def plot_accuracies_by_training_size_and_user_scatter(results_df, user_id, training_sizes = [10,20,30,40,50,60,70,80,90,100], \
														sampling_method='random'):
	models_dict = get_performance_means(results_df, sampling_method=sampling_method)
	
	traces = []
	for model, data in models_dict.items():
		if model != "impersonal":
			#print("%s" % model)
			means = []
			for ts in training_sizes:
				mean_accuracy = data['training_sizes'][ts][user_id]
				#print("\t%s personal : %s" % (ts, mean))
				means.append(mean_accuracy)
			#print(means)
			#print(training_sizes)
			trace = go.Scatter(y=means,
							   x=training_sizes,
							   name=model)
			traces.append(trace)
		else:
			#print("%s" % model)
			mean = data[user_id]
			trace = go.Scatter(y=[mean]*len(training_sizes),
							   x=training_sizes,
							   name=model)
			traces.append(trace)
			#print("\t%s personal : %s" % (0, mean))
		
	layout = go.Layout(showlegend=True,
			yaxis=dict(title="Accuracy in %", range=[0,1.0]),
			xaxis=dict(title="Amount of training data or training burden to the user"))
	
	fig = go.Figure(data=traces, layout=layout)
	return fig

def plot_accuracies_by_training_size_scatter(results_df, training_sizes = [10,20,30,40,50,60,70,80,90,100], \
											aggregation_method=np.mean, sampling_method='random'):
	models_dict = get_performance_means(results_df, sampling_method=sampling_method)
	traces = []
	for model, data in models_dict.items():
		if model != "impersonal":
			#print("%s" % model)
			means = []
			for ts in training_sizes:
				mean = aggregation_method([v for v in data['training_sizes'][ts].values()])
				#print("\t%s personal : %s" % (ts, mean))
				means.append(mean)
			#print(means)
			#print(training_sizes)
			trace = go.Scatter(y=means,
							   x=training_sizes,
							   name=model)
			traces.append(trace)
		else:
			#print("%s" % model)
			mean = aggregation_method([v for v in data.values()])
			trace = go.Scatter(y=[mean]*len(training_sizes),
				   x=training_sizes,
				   name=model)
			traces.append(trace)
			#print("\t%s personal : %s" % (0, mean))
		
	layout = go.Layout(showlegend=True,
			yaxis=dict(title="Accuracy in %", range=[0,1.0]),
			xaxis=dict(title="Amount of training data or training burden to the user"))
	
	fig = go.Figure(data=traces, layout=layout)
	return fig

def percentile_50(arr):
	return np.percentile(arr, 50)
def percentile_25(arr):
	return np.percentile(arr, 25)
def percentile_20(arr):
	return np.percentile(arr, 20)
def percentile_10(arr):
	return np.percentile(arr, 10)
