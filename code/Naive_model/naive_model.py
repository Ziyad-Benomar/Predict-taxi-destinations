#---------------------------------------------------------------------
## Import libraries
#----------------------------------------------------------------------
import pandas as pd
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from haversine import haversine
from tqdm import tqdm 
tqdm.pandas()



#---------------------------------------------------------------------
## The Haversine distance
#----------------------------------------------------------------------
from haversine import haversine as hv_lat_lon
# In our dataset, points are given as couples (longitude, latitude), while hv_lat_lon needs points in the reversed order
# (latitude, longitude). Therfeore we need to reverse the coordinates when computing the distance. We define the function:
def haversine(M, P) :
    return hv_lat_lon(M[::-1], P[::-1])





#---------------------------------------------------------------------
## Functions for the bearing
#----------------------------------------------------------------------
def get_bearing(M1, M2):
    lon1, lat1 = M1 
    lon2, lat2 = M2
    dLon = (lon2 - lon1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y) # in radians
    return brng

def get_traj_bearing(traj) :
    return get_bearing(traj[0], traj[-1])

def sin_error(x) :
    return abs(np.sin(x/2))





#---------------------------------------------------------------------
## Predicting functions
#----------------------------------------------------------------------

# very naive prediction (last available point)
#----------------------------------------------------------------------
def naive_pred(row):
    return row.POLYLINE[-1]


# prediction = average point of centroids having the "good" bearing 
#----------------------------------------------------------------------
def bearing_pred(row, cluster_centers, min_len=40, threshold=0.5) :
    """
    traj_cut : a prfix of a trajectory
    cluster centers
    """
    traj = row.POLYLINE
    if len(traj) < min_len :
        return traj[-1]
    
    M = traj[0]
    traj_bearing = get_traj_bearing(traj)
    close_centers = []
    for center in cluster_centers :
        bearing = get_bearing(M, center)
        err = sin_error(bearing - traj_bearing)
        if err < threshold :
            close_centers.append(center)
    close_centers = np.array(close_centers)
    if len(close_centers) == 0 :
        return traj[-1]
    lon_pred = close_centers[:,0].mean()
    lat_pred = close_centers[:,1].mean()
    return lon_pred, lat_pred



# prediction = weighted average point of centroids having the 
# "good" bearing, where weights depend on the distance from the
# departure point
#----------------------------------------------------------------------
def softmax(x, alpha=1) :
    expx = np.exp(-alpha*x)
    sum_expx = expx.sum()
    return expx/sum_expx

def compute_length_weights(M, close_centers, expected_length, alpha=1) :
    hav_distances =  np.zeros(len(close_centers))
    for i in range(len(hav_distances)) :
        center = close_centers[i]
        hav_distances[i] = haversine(M, center)
    diff = np.abs(hav_distances - expected_length)
    return softmax(diff, alpha)

from datetime import datetime

# Week of the year
def get_week(tstamp) :
    return datetime.fromtimestamp(tstamp).date().isocalendar()[1] - 1 # -1 to have indices starting from 0

# Day of the week
day_ix = {"Monday":0, "Tuesday":1, "Wednesday":2,
          "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
def get_day_of_week(tstamp) :
    day = datetime.fromtimestamp(tstamp).strftime("%A")
    return day_ix[day]

# Quarter hour
def get_quarter_hour(tstamp) :
    hour = datetime.fromtimestamp(tstamp).hour
    minute = datetime.fromtimestamp(tstamp).minute
    return 4*hour + minute//15


def bearing_length_pred(row, cluster_centers, lengths, alpha=1, threshold=0.86, min_len=370) :
    """
    traj_cut : a prfix of a trajectory
    cluster centers
    """
    traj = row.POLYLINE
    if len(traj) < min_len :
        return traj[-1]

    tstamp = row.TIMESTAMP
    week = get_week(tstamp)
    day = get_day_of_week(tstamp)
    quarter_hour = get_quarter_hour(tstamp)
    expected_length = lengths[week, day, quarter_hour]
    
    M = traj[0]
    traj_bearing = get_traj_bearing(traj)
    close_centers = []
    # get cluster centers that are in the direction of the trajectory
    for center in cluster_centers:
        bearing = get_bearing(M, center)
        err = sin_error(bearing - traj_bearing)
        if err < threshold :
            close_centers.append(center)
    if len(close_centers) == 0 :
        return traj[-1]
    # Compute their weighted average 
    close_centers = np.array(close_centers)
    weights = compute_length_weights(M, close_centers, expected_length, alpha=alpha)
    lon_pred = np.average(close_centers[:,0], weights=weights)
    lat_pred = np.average(close_centers[:,1], weights=weights)
    return lon_pred, lat_pred




#----------------------------------------------------------------------
# Compute the loss on a test dataframe given a prediction function 
#----------------------------------------------------------------------

def compute_loss(test_df, test_sol, pred_fctn=naive_pred, params={}) :
    loss = []
    for i in range(test_df.shape[0]):
        params['row'] = test_df.iloc[i]
        arrival_pred = pred_fctn(**params)
        arrival = [test_sol.iloc[i].LONGITUDE, test_sol.iloc[i].LATITUDE]
        loss.append(haversine(arrival, arrival_pred))
    return loss




#----------------------------------------------------------------------
# Running the improved naive model (weighted average)
#----------------------------------------------------------------------
# import the test set
print("importing the test set...", end='')
test_df = pd.read_csv("../Data/test.csv")
test_sol = pd.read_csv("../Data/solution_fixed.csv")
test_df["POLYLINE"] = test_df["POLYLINE"].progress_apply(lambda x: eval(x))
print("[ok]")

# import the cluster centers
print("importing the cluster centers...", end='')
cluster_centers = pd.read_csv("cluster_centers.csv")
cluster_centers = np.array([[cluster_centers.iloc[i].x, cluster_centers.iloc[i].y] for i in range(cluster_centers.shape[0])])
print("[ok]")

# compute the L_hav matrix
def compute_length(traj) :
    Mi = traj[0]
    Mf = traj[-1]
    return haversine(Mi, Mf)

print("importing the clean training dataset...", end='')
df = pd.read_csv("../CleanData/train_clean.csv")
df['POLYLINE'] = df['POLYLINE'].progress_apply(lambda x: eval(x))
df = df.loc[df["POLYLINE"].apply(lambda x : len(x)>3)]
df["LENGTH"] = df["POLYLINE"].progress_apply(compute_length)
print("[ok]")

# mean lengths computed for each (week, day, quarter_hour) tuple
print("computing L_hav...", end='')
lengths = np.zeros((52,7,96))
count = np.zeros((52,7,96))

for i in range(df.shape[0]) :
    row = df.iloc[i]
    lengths[row.week, row.day, row.quarter_hour] += row.LENGTH
    count[row.week, row.day, row.quarter_hour] += 1
    
lengths /= count

# remove nan values:
# we replace them by the mean value of lengths
mean_length = df.LENGTH.mean()
for week in range(52) :
    for day in range(7) :
        for quart_hour in range(96) :
            if np.isnan(lengths[week, day, quart_hour]) :
                lengths[week, day, quart_hour] = mean_length

# make the dependency w.r.t quarter_hour smoother
# to avoid overfitting
lengths_smooth = np.copy(lengths)
for s in range(5) :
    for week in range(52) :
        for day in range(7) :
            for quart_hour in range(1,95) :
                curr = lengths_smooth[week, day, quart_hour]
                prev = lengths_smooth[week, day, quart_hour - 1]
                next = lengths_smooth[week, day, quart_hour + 1]
                lengths_smooth[week, day, quart_hour] = curr/2 + prev/4 + next/4
print("[ok]")

# Running the model
params = {
    "cluster_centers": cluster_centers,
    "lengths": lengths_smooth,
    "threshold": 0.86,
    "min_len": 370,
    "alpha": 3.05
}
print("Running the improved naive model")
loss = compute_loss(test_df, test_sol, pred_fctn=bearing_length_pred, params=params)
print("the loss with the improved naive model is ", np.mean(loss))