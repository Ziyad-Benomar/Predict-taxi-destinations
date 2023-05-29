import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
tqdm.pandas()
import ast
from scipy import stats
import seaborn as sns
import os



#---------------------------------------------------------------------
## Encoding Dics For the Metadata
#----------------------------------------------------------------------
# Encoding dictionaries

import json

def create_encoders() :
    # Import the columns to encode
    data = pd.read_csv('Data/train.csv', usecols=["ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID"])

    # Create an encoding for ORIGIN_CALL
    origin_call_set = set(data["ORIGIN_CALL"].unique())
    origin_call_ix = {str(id): i for i, id in enumerate(origin_call_set)}

    # Create an encoding for ORIGIN_STAND
    origin_stand_set = set(data["ORIGIN_STAND"].unique())
    origin_stand_ix = {str(id): i for i, id in enumerate(origin_stand_set)}

    # Create an encoding for TAXI_ID
    taxi_id_set = set(data["TAXI_ID"].unique())
    taxi_id_ix = {str(id): i for i, id in enumerate(taxi_id_set)}

    # Create an encoding for day
    day_ix = {"Monday":0, "Tuesday":1, "Wednesday":2,
                "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}

    # !!! The keys are STR !!! (to save properly as json : nan is not an acceptable type)

    # Save the en encoders
    with open('Encoders/encoders.json', 'w') as file:
        dics_to_json = [origin_call_ix, origin_stand_ix, taxi_id_ix, day_ix]
        json.dump(dics_to_json, file)
        file.close()

create_encoders()





#---------------------------------------------------------------------
## Auxilary Functions for Cleaning the Training Dataset
#----------------------------------------------------------------------


from haversine import haversine
from itertools import compress
from datetime import datetime
dt_object = datetime.fromtimestamp(1372636858)
import random

# Auxilary functions

# Convert POLYLINE from str to list
def get_polyline(df) :
    df['POLYLINE'] = df['POLYLINE'].progress_apply(lambda x: eval(x))
    return df

def discrete_speed(traj):
    """for a trajectory, estimate the speed between each gps value in km/h"""
    speed = []
    for i in range(1, len(traj)):
        speed.append(haversine(traj[i-1], traj[i])*4*60)
    return speed

def remove_redundant(traj, speed_list,  threshold_min = 1):
    filter_list = (list((map(lambda x: x>threshold_min, speed_list))))
    res = list(compress(traj, [True]+filter_list))
    return res

def get_outliers(x, y, threshold = 6):
    # Return the indices of the outliers
    z_score_x = (np.abs(stats.zscore(x)))
    z_score_y = (np.abs(stats.zscore(y)))
    
    indices_x = np.where((z_score_x>threshold))
    indices_y = np.where((z_score_y>threshold))
    
    indices = list(set().union(list(np.ravel(indices_x)), list(np.ravel(indices_x))))
    
    return indices

def remove_outliers(df) :
    df['SPEED'] = df['POLYLINE'].progress_apply(discrete_speed)
    threshold_max = 200
    df = df.loc[df['SPEED'].apply(lambda x: max(x)<threshold_max)]
    df['POLYLINE'] = df.progress_apply(lambda x: remove_redundant(x['POLYLINE'], x['SPEED']), axis = 1)

    # Arrival outliers
    arrival = df['POLYLINE'].apply(lambda x: x[-1])
    indices_arr = get_outliers(arrival.apply(lambda x: x[0]), arrival.apply(lambda x: x[1]))

    # Departure outliers
    departure = df['POLYLINE'].apply(lambda x: x[0])
    indices_dep = get_outliers(departure.apply(lambda x: x[0]), departure.apply(lambda x: x[1]))

    # Removing the outliers
    indices = list(set().union(list(np.ravel(indices_dep)), list(np.ravel(indices_arr))))
    df.drop(df.index[indices], axis = 0, inplace = True)

    # Remove the SPEED feature
    del df["SPEED"]

    return df



# Extract week, day of the week and quarter hour from timestamp

# Week of the year
def get_week(tstamp) :
    return datetime.fromtimestamp(tstamp).date().isocalendar()[1] - 1 # -1 to have indices starting from 0

# Day of the week
def get_day_of_week(tstamp) :
    return datetime.fromtimestamp(tstamp).strftime("%A")

# Quarter hour
def get_quarter_hour(tstamp) :
    hour = datetime.fromtimestamp(tstamp).hour
    minute = datetime.fromtimestamp(tstamp).minute
    return 4*hour + minute//15

# Function to extract the time features and remove TIMESTAMP 
def get_time_features(df) :
    df['week'] = df['TIMESTAMP'].progress_apply(lambda x: get_week(x))
    df['day'] = df['TIMESTAMP'].progress_apply(lambda x: get_day_of_week(x))
    df['quarter_hour'] = df['TIMESTAMP'].progress_apply(lambda x: get_quarter_hour(x))
    return df.drop(["TIMESTAMP"], axis=1)


# Encoding the metadf

def get_encoding(dict, key) :
    # If the key is in the dict, return the corresponding value
    if key in dict : 
        return dict[key]
    # If not, associate to the 'nan' key if it exists
    if 'nan' in dict :
        return dict['nan']
    # Otherwise, return a random value
    return random.choice(list(dict.values()))


def my_encode(df) :
    """
    Encodes the features ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, day
    """
    with open('Encoders/encoders.json', 'rb') as file:
        origin_call_ix, origin_stand_ix, taxi_id_ix, day_ix = eval(file.read()) # file.read() reads a str
        # keys must be converted to str
        df["ORIGIN_CALL"] = df["ORIGIN_CALL"].progress_apply(lambda x: get_encoding(origin_call_ix,str(x)))
        df["ORIGIN_STAND"] = df["ORIGIN_STAND"].progress_apply(lambda x: get_encoding(origin_stand_ix,str(x)))
        df["TAXI_ID"] = df["TAXI_ID"].progress_apply(lambda x: get_encoding(taxi_id_ix,str(x)))
        df["day"] = df["day"].apply(lambda x: day_ix[x])
        df["CALL_TYPE"] = df["CALL_TYPE"].apply(lambda x: ord(x) - ord('A'))
        file.close()
    return df

#---------------------------------------------------------------------
## Main Function for Cleaning the Training Dataset
#----------------------------------------------------------------------

# The cleaning function
def clean(df) :

    df = df[df["MISSING_DATA"] == False]    
    del df["TRIP_ID"], df["DAY_TYPE"], df["MISSING_DATA"]

    df = get_polyline(df)
    df = df.loc[df['POLYLINE'].apply(lambda x: len(x)>1)]
    df = remove_outliers(df)
    df = get_time_features(df)
    df = my_encode(df)

    return df


#---------------------------------------------------------------------
## Cleaning the Training Dataset and Saving it!
#----------------------------------------------------------------------
# Create directory CleanData
if not os.path.exists("CleanData"):
    os.makedirs("CleanData")

# Remove the train_ready if it already exists (from a previous run of this code)
if os.path.exists('CleanData/train_clean.csv'):
    os.remove('CleanData/train_clean.csv')

# Since we cannot load the whole dataset at once (too large), we will read sequences of 500000 rows and clean them one after the other 
for i in range(4) :
    print('\n'+'-'*100 +'\ni = ',i,'...' )
    # Load data
    data = pd.read_csv('Data/train.csv', skiprows=range(1,i*500000), nrows=500000)
    # Clean the data
    data = clean(data)
    # Save the data
    data.to_csv("CleanData/train_clean.csv", mode='a', header=(i==0), index=False)




#---------------------------------------------------------------------
## Shuffling the training data
#----------------------------------------------------------------------

# Load the whole dataset, shuffle it and save it again !
data = pd.read_csv("CleanData/train_clean.csv")
data = data.sample(data.shape[0])
data.to_csv("CleanData/train_clean.csv", index=False)

print(f"Total number of rows : ",data.shape[0])