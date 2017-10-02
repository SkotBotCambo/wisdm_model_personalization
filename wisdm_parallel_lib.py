import sys
import pandas as pd
import time

temporary_dataframes_locations = "/home/sac086/wisdm_model_personalization/datasets/WISDM_v2/temporary_user_dataframes/"

def assign_segments_by_user(user_id, user_df, windowSize=10):
	td = pd.Timedelta(str(windowSize) + ' seconds')
	segment_col = pd.Series(index=user_df.index, dtype='int32')

	print("Segmenting data for user #%s" % user_id)
	print("\t %s has %s rows" % (user_id, len(user_df)))
	start_time = time.time()

	last_timestamp = user_df['timestamp'].max()
	beginning_of_segment = user_df['timestamp'].min()
	end_of_segment = beginning_of_segment + td
	segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
					  (user_df['timestamp'] < end_of_segment)]
	
	segment_id = 0
	while beginning_of_segment < last_timestamp:
		#print("Segment #%s" % segment_id)
		if len(segment_df) < 1:
			# set beginning of segment to reflect next timestamp
			beginning_of_segment = user_df[user_df['timestamp'] > beginning_of_segment]['timestamp'].min()
			end_of_segment = beginning_of_segment + td
			segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
					  (user_df['timestamp'] < end_of_segment)]
			continue
		segment_col.loc[segment_df.index] = segment_id
		#print("\tupdated segment column")
		# make updates for next iteration
		segment_id += 1
		beginning_of_segment = end_of_segment
		end_of_segment = beginning_of_segment + td
		segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
					  (user_df['timestamp'] < end_of_segment)]
	finish_time = time.time()
	print("\tfinished in %s seconds" % (finish_time - start_time))
	user_df['segment_id'] = segment_col
	return user_df

def segment_users(user_dfs):
	for (user_id, user_df) in user_dfs:
		user_df = assign_segments_by_user(user_id, user_df)
		file_loc_name = temporary_dataframes_locations + user_id+"_raw_segmented.pickle"
		print("\tSaving dataframe...")
		user_df.to_pickle(file_loc_name)
		print("\tSaved.")
		print("")