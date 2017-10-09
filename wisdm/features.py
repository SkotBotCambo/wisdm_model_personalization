import numpy as np
import pandas as pd

bin_ranges = list(np.arange(-2.5, 20, 2.5))
bin_ranges = [-np.inf] + bin_ranges
bin_ranges = bin_ranges + [np.inf]

feature_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
				 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9',
				 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9',
				 'XAVG', 'YAVG', 'ZAVG',
				 'XPEAK', 'YPEAK', 'ZPEAK',
				 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
				 'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV',
				 'RESULTANT']

def fill_bins(segment_df):
    counts_x, _ = np.histogram(segment_df['x-acc'], bins=bin_ranges, density=False)
    counts_y, _ = np.histogram(segment_df['y-acc'], bins=bin_ranges, density=False)
    counts_z, _ = np.histogram(segment_df['z-acc'], bins=bin_ranges, density=False)
    return counts_x, counts_y, counts_z

def get_avg(segment_df):
    xavg = segment_df['x-acc'].mean()
    yavg = segment_df['y-acc'].mean()
    zavg = segment_df['z-acc'].mean()
    return xavg, yavg, zavg

def get_peak_times(segment_df):
    axes = ['x-acc', 'y-acc', 'z-acc']
    
    peak_feature_values = {axis : None for axis in axes}
    for axis in axes:
        sorted_indeces = segment_df[axis].argsort()
        max_peak = segment_df[axis].iloc[sorted_indeces.iloc[-1]]
        threshold = max_peak * 0.9
        peaks_df = segment_df[segment_df[axis] > threshold]
        if len(peaks_df) < 3:
            peaks_df = segment_df.iloc[sorted_indeces.iloc[-3:]]
        
        peaks_df.sort_values('timestamp', ascending=False, inplace=True)

        peak_diffs = []
        #print(len(peaks_df))
        iter_peaks = peaks_df['timestamp'].iteritems()
        _, last_ts = next(iter_peaks)

        for _, pt in peaks_df['timestamp'].iteritems():
            peak_diffs.append(pt.timestamp() - last_ts.timestamp())
            last_ts = pt
        peak_feature_values[axis] = np.mean(peak_diffs)

    return peak_feature_values['x-acc'], peak_feature_values['y-acc'], peak_feature_values['z-acc']

def get_absdev(segment_df):
    x_mean = segment_df['x-acc'].mean()
    y_mean = segment_df['y-acc'].mean()
    z_mean = segment_df['z-acc'].mean()
    
    x_absdev = np.mean([np.absolute(x_mean - x) for x in segment_df['x-acc']])
    y_absdev = np.mean([np.absolute(y_mean - y) for y in segment_df['y-acc']])
    z_absdev = np.mean([np.absolute(z_mean - z) for z in segment_df['z-acc']])
    
    return x_absdev, y_absdev, z_absdev

def get_sd(segment_df):
    x_std = segment_df['x-acc'].std()
    y_std = segment_df['y-acc'].std()
    z_std = segment_df['z-acc'].std()
    
    return x_std, y_std, z_std

def get_resultant(segment_df):
    values = []
    
    for ind, row in segment_df.iterrows():
        sum_val = (row['x-acc']**2) + (row['y-acc'] ** 2) + (row['z-acc'] ** 2)
        values.append(sum_val)
    return np.mean(values)