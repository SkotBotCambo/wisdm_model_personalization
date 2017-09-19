import pandas as pd
import numpy as np
import datetime as dt
from time import time

raw_data_location = "./datasets/WISDM_v2/all_raw_data.dataframe.pickle"
print("loading data...")
raw_df = pd.read_pickle(raw_data_location)
print("data loaded.")

user_ids = raw_df['user'].unique()

def assign_segments(raw_df, samplingRate=20, windowSize=10, window_movement='discrete'):
    users = raw_df['user'].unique()
    segment_id = 0
    
    td = pd.Timedelta(str(windowSize) + ' seconds')
    
    segment_col = pd.Series(index=raw_df.index, dtype="int32")
    
    for user_id in users:
        print("Segmenting data for user #%s" % user_id)
        start_time = time()
        user_df = raw_df[raw_df['user'] == user_id]
        
        last_timestamp = user_df['timestamp'].max()
        beginning_of_segment = user_df['timestamp'].min()
        end_of_segment = beginning_of_segment + td
        segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
                          (user_df['timestamp'] < end_of_segment)]

        while beginning_of_segment < last_timestamp:
            if len(segment_df) < 1:
                # set beginning of segment to reflect next timestamp
                beginning_of_segment = user_df[user_df['timestamp'] > beginning_of_segment]['timestamp'].min()
                end_of_segment = beginning_of_segment + td
                segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
                          (user_df['timestamp'] < end_of_segment)]
                continue
            segment_col.loc[segment_df.index] = segment_id
            
            # make updates for next iteration
            segment_id += 1
            beginning_of_segment = end_of_segment
            end_of_segment = beginning_of_segment + td
            segment_df = user_df[(user_df['timestamp'] > beginning_of_segment) & \
                          (user_df['timestamp'] < end_of_segment)]
        finish_time = time()
        print("finished in %s seconds" % (finish_time - start_time))
    
    raw_df['segment_id'] = segment_col
    return raw_df

print("assigning segments...")
raw_df = assign_segments(raw_df)
raw_df.to_pickle(raw_data_location)