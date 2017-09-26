
import ipyparallel as ipp
import pandas as pd
import datetime as dt
import time
import sys
c = ipp.Client()
dview = c[:]
dview.block=False

results = dview.execute('import sys; sys.path.append("/home/sac086/wisdm_model_personalization/")')

print("Setting up parallelization...")
with dview.sync_imports():
	import wisdm_parallel_lib



raw_data_location = "./datasets/WISDM_v2/all_raw_data.dataframe.pickle"
print("loading data...")
raw_df = pd.read_pickle(raw_data_location)
print("data loaded.")

user_ids = raw_df['user'].unique()

def wait_watching_stdout(ar, dt=1, truncate=1000):
	while not ar.ready():
		stdouts = ar.stdout
		if not any(stdouts):
			continue
		# clear_output doesn't work in plain terminal / script environments
		print('-' * 30)
		print("%.3fs elapsed" % ar.elapsed)
		print("")
		for eid, stdout in zip(ar._targets, ar.stdout):
			if stdout:
				print("[ stdout %2i ]\n%s" % (eid, stdout[-truncate:]))
		sys.stdout.flush()
		time.sleep(dt)
	if ar.successful():
		print("successfully finished...")
	else:
		print("getting problem...")
		ar.get()

print("There are %s users" % len(user_ids))
print("splitting dataframes by user...")
user_dfs = []
for user_id in user_ids:
	user_df = raw_df[raw_df['user'] == user_id]
	user_dfs.append((user_id, user_df))
print("finished splitting dataframes")
print("Starting segmentation on cluster")
#scatter_result = dview.scatter("user_ids", unfinished_user_ids)
#dview.execute(command)
scatter_result = dview.scatter("user_dfs", user_dfs)
command = "wisdm_parallel_lib.segment_users(user_dfs)"

ar = dview.execute(command)
#ar = dview.apply_async(segment_users, user_dfs)
wait_watching_stdout(ar)
print("Finished segmentation")