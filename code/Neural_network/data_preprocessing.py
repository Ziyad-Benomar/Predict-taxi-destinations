#---------------------------------------------------------------------
## Importing Libraries
#----------------------------------------------------------------------
import pandas as pd
import numpy as np
from tqdm import tqdm 
tqdm.pandas()

#---------------------------------------------------------------------
## Process data for the neural net
### Auxilary functions
#----------------------------------------------------------------------
# Auxilary functions

# Convert POLYLINE from str to list
def get_polyline(df) :
    df['POLYLINE'] = df['POLYLINE'].progress_apply(lambda x: eval(x))
    return df

def k_first_last(L,k) :
    """
    Input  : a list L and an integer k
    Output : a 2k list of the k first and k last elements of L
             if L is too small, the first element is repeated
    """
    if len(L) >= 2*k :
        return L[:k] + L[-k:]
    return [L[0]]*(2*k - len(L)) + L

def random_prefix(L) :
    """
    Input  : a list L
    Output : a prefix of the trajectory with random size
    """
    if len(L) == 1 : return L
    r = np.random.randint(1,len(L))
    return L[:r+1]


# In the article, all the prefixes of all trajectories are considered
# but with limited computed ressources, we will instead choose a number M of random prefixes
# for each trajectories. This comes to copying the dataframe M times and keeping a random
# prefix of the trajectory in each line

def duplicate_lines(df, M) :
    """
    Input  : dataframe df and an integer M
    Output : a concatenation of M copies of df
    """
    frames = []
    for _ in range(M) :
        frames.append(df.copy())
    return pd.concat(frames)


def get_dest(df) :
    # First save the destination point
    # We need to keep Y a feature of df for now because we will duplicate some rows in the following
    df["x_DEST"] = df["POLYLINE"].apply(lambda L : L[-1][0])
    df["y_DEST"] = df["POLYLINE"].apply(lambda L : L[-1][1])
    return df

# Data scaling functions
# SHOULD WE SCALE THE DATA ????
def scale(x, xmin, xmax) :
    return (x-xmin)/(xmax-xmin)

def save_scaling_factors(xmin, xmax, ymin, ymax) :
    with open('Encoders/scaling.txt', 'w') as file:
        for val in [xmin, xmax, ymin, ymax] :
            file.write(str(val)+'\n')
        file.close()

#---------------------------------------------------------------------
### The main preprocessing function
#----------------------------------------------------------------------
def process(df, M=4, k=5) :
    # The feature CALL_TYPE will not be used
    del df["CALL_TYPE"]
    # Transorm POLYLINE from str to list
    df = get_polyline(df)
    # Extract the arrival points
    df = get_dest(df)

    # If a trajectory contains less than 2k points, there is no need to slice it,
    # because this will only degrade the information we have, given that we will
    # artificially make all trajectories of length 2k using the function k_first_last.
    df_short = df.loc[df['POLYLINE'].apply(lambda x: len(x)<=2*k)]
    df_long = df.loc[df['POLYLINE'].apply(lambda x: len(x)>2*k)]
    # duplicate only trajectories that are longer than 2k and randomly slice them
    df_long = duplicate_lines(df_long, M)
    df_long["POLYLINE"] = df_long["POLYLINE"].apply(lambda L : random_prefix(L))
    # Concatenate the two dataframes and make all the trajectories of length 2k
    df = pd.concat([df_short, df_long])
    df["POLYLINE"] = df["POLYLINE"].apply(lambda L : k_first_last(L,k))

    # Compute scaling boundaries and save them 
    # Remark : the scaling we make will not guarantee us to have values in [0,1] because
    # the max and min values are only computed over the arrival points set, but it give the data 
    # the scale we want Θ(1) --------------
    
    #xmin, xmax = df["x_DEST"].min(), df["x_DEST"].max()
    #ymin, ymax = df["y_DEST"].min(), df["y_DEST"].max()
    #save_scaling_factors(xmin, xmax, ymin, ymax)

    # Save the POLYLINE points into 4k numerical features x_0,..,x_2k, y_0,..,y_2k
    for i in range(2*k) :
        #col_nonscaled = df["POLYLINE"].apply(lambda L : L[i][0])
        #df["x" + str(i)] = col_nonscaled.apply(lambda x: scale(x,xmin,xmax))
        df["x" + str(i)] = df["POLYLINE"].apply(lambda L : L[i][0])
    for i in range(2*k) :
        #col_nonscaled = df["POLYLINE"].apply(lambda L : L[i][1])
        #df["y" + str(i)] = col_nonscaled.apply(lambda x: scale(x,ymin,ymax))
        df["y" + str(i)] = df["POLYLINE"].apply(lambda L : L[i][1])
    del df["POLYLINE"]
    
    return df

#---------------------------------------------------------------------
## Processing the data and saving it
#----------------------------------------------------------------------
import os

M = 8
k = 5

# Remove the train_ready if it already exists (from a previous run of this code)
if os.path.exists('../CleanData/train_ready.csv'):
    os.remove('../CleanData/train_ready.csv')

# Since we cannot load the whole dataset at once (too large), we will read sequences of 500000 rows and clean them one after the other 
for i in range(4) :
    print('\n'+'-'*100 +'\ni = ',i,'...' )
    # Load data
    data = pd.read_csv('../CleanData/train_clean.csv', skiprows=range(1,i*500000), nrows=500000)
    # Process the data
    data = process(data,M=M,k=k)
    # Save the data
    data.to_csv("../CleanData/train_ready.csv", mode='a', header=(i==0), index=False)

#---------------------------------------------------------------------
## Shuffle the data and save it again
#----------------------------------------------------------------------
# Load the whole dataset, shuffle it and save it again !
data = pd.read_csv("../CleanData/train_ready.csv")
data = data.sample(len(data))
data.to_csv("../CleanData/train_ready.csv", index=False)

print(f"Total number of rows : ",data.shape[0])