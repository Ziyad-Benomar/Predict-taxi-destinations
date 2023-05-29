#---------------------------------------------------------------------
## Import libraries
#----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time






#---------------------------------------------------------------------
## Functions for Loading Data
#---------------------------------------------------------------------

# Load a block of data

def load_datablock(i) :
    """
    Loads the i-th block of data, shuffles it and returns it
    """
    filename = "TrainData/train_ready" + str(i) +".csv"
    df = pd.read_csv(filename)
    return df.sample(df.shape[0])


# Load data, shuffle it and save it to blocks 

def shuffle_data(num_blocks, block_size=32000) :
    """
    Reads the clean data, shuffles it, and then saves it into several data blocks
    allowing to read the data faster when training the neural network
    """
    ti = time.time()
    # Load data
    data = pd.read_csv("../CleanData/train_ready.csv")
    # Shuffle it
    data = data.sample(len(data))
    # Save it into blocks
    for i in range(num_blocks) :
        filename = "TrainData/train_ready" + str(i) +".csv"
        data.iloc[i*block_size:(i+1)*block_size].to_csv(filename, index=False)
    tf = time.time()
    return tf-ti
    

# Transform data to tensors
def dataframe_to_tensor(df, cuda=True) :
    """
    Converts a dataframe to torch.tensor
    Moves it to the GPU if cuda == True
    """
    df_tensor = torch.from_numpy(df.to_numpy()).float()
    if cuda : 
        df_tensor = df_tensor.cuda()
    return df_tensor


# WARNING : The order of the columns in the metadata is important, because the neural net reads them in this order
metacolumns = ["ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID", "week", "day", "quarter_hour"]

def get_minibatch(df, i, batch_size=32) :
    """
    df must necessarily have a size not smaller than (i+1)*batch_size
    This function returns three tensors ready to be used by the neural network : meta, traj, dest
    """
    # Get a batch dataframe (the shuffling was made before on the entire block)
    df_batch = df.iloc[i*batch_size:(i+1)*batch_size]

    # Extract the destination point
    dest = dataframe_to_tensor(df_batch[["x_DEST","y_DEST"]])
    df_batch = df_batch.drop(["x_DEST","y_DEST"], axis=1)

    # Get meta data and trajectory data
    meta = dataframe_to_tensor(df_batch[metacolumns]).long()
    traj = dataframe_to_tensor(df_batch.drop(metacolumns, axis=1))
    
    return meta, traj, dest






#---------------------------------------------------------------------
## The Neural Network and the LOss Function
#---------------------------------------------------------------------

# Auxilary function : get the size of embeddings
def get_metadata_num_values() :
    with open('Encoders/encoders.json', 'rb') as file:
        origin_call_ix, origin_stand_ix, taxi_id_ix, day_ix = eval(file.read())
        # origin_call, origin_stand, taxi_id, week, day, quarter_hour
        return len(origin_call_ix), len(origin_stand_ix), len(taxi_id_ix), 52, 7, 96 

class MyNet(nn.Module):

    def __init__(self, cluster_centers, emb_dims=[10]*5, k=5, hidden_layer_size=500):
        """
        value_counts : number of distinct values of each metadata feature
                      - origin_call
                      - origin_stand
                      - taxi_id
                      - week
                      - day
                      - hour
        emb_dims     : embeding dimensions of each metadata feature 
        k            : 2k is the number of points we kept from the trajectory
        cluster_centers : a torch tensor with 2 columns. Each row gives the coordinates
                          of a cluster center.
                          we assume that a clustering has been made using mean-shift.
                       
        """
        super(MyNet, self).__init__()
        self.cluster_centers = cluster_centers

        # Embedding -------------------------------------------------
        origin_call_count, origin_stand_count, taxi_id_count, week_count, day_count, quart_hour_count = get_metadata_num_values()
        origin_call_emdim, origin_stand_emdim, taxi_id_emdim, week_emdim, day_emdim, quart_hour_emdim = emb_dims
        metadata_emb_size = np.array(emb_dims).sum()

        max_norm = 1
        self.emb_origin_call = nn.Embedding(origin_call_count, origin_call_emdim, max_norm=max_norm)
        self.emb_origin_stand = nn.Embedding(origin_stand_count, origin_stand_emdim, max_norm=max_norm)
        self.emb_taxi_id = nn.Embedding(taxi_id_count, taxi_id_emdim, max_norm=max_norm)
        self.emb_week = nn.Embedding(week_count, week_emdim, max_norm=max_norm)
        self.emb_day = nn.Embedding(day_count, day_emdim, max_norm=max_norm)
        self.emb_quart_hour = nn.Embedding(quart_hour_count, quart_hour_emdim, max_norm=max_norm)
        
        # Linear-Relu-Softmax ----------------------------------------
        input_size = metadata_emb_size + 4*k #k first and last points, each having 2 coords
        num_clusters = cluster_centers.shape[0]
        self.layers = nn.Sequential( # hidden layer with 500 ReLU neurons, followed by softmax
            nn.Linear(input_size,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,num_clusters),
            nn.ReLU(),
            nn.Softmax()
        )

    def embed_metadata(self,meta) :
        """
        meta : a tensor with the metadata columns
        returns concatenation of embeddings of metadata
        """
        origin_call,origin_stand,taxi_id,week,day,quart_hour = meta[:,0],meta[:,1],meta[:,2],meta[:,3],meta[:,4],meta[:,5]
        embed1 = self.emb_origin_call(origin_call)
        embed2 = self.emb_origin_stand(origin_stand)
        embed3 = self.emb_taxi_id(taxi_id)
        embed4 = self.emb_week(week)
        embed5 = self.emb_day(day)
        embed6 = self.emb_quart_hour(quart_hour)
        return torch.cat((embed1,embed2,embed3,embed4,embed5,embed6), 1)

    def compute_centroids(self,p) :
        """
        Input  : a tensor where each line is a vector of positive weights with sum =1
        Output : Centroids of the cluster centers with the given weights in each line
                 the number of rows of the output is the same as the input's
        """
        return p @ self.cluster_centers
    
    def forward(self, meta, traj):
        """
        meta : metadata columns
        traj : 4k columns showing the first and last k points of each trajectory
        """
        # Embedding 
        embed = self.embed_metadata(meta)
        
        # Since the traj data is (approximatively) scaled in [0,1], we will do the same for the embedding
        # in the construction we constrained the max norm = 1, now we will make it in the interval [0,1]
        #embed = (embed+1)/2
        
        # Concatenate the embed tensor with the trajectory tensor
        x = torch.cat((embed, traj), 1)
        # Liner-ReLu-Softmax layer, the output is a tensor of the weights associated with the clusters 
        x = self.layers(x)
        # Compute the centroids with the given weights
        x = self.compute_centroids(x)
        return x



# Equirectangular distance
# x=longitude, y=latitude

relu = nn.ReLU()

def loss_equirec(Y1,Y2) :
    """
    Input  : 2 tensors of the same size, each having two columns (latitude, longitude)
    Output : Mean equirectangular distance between the points of Y1 and the points of Y2
    """
    lng1, lat1 = Y1[:,0], Y1[:,1] #
    lng2, lat2 = Y2[:,0], Y2[:,1] #
    dlng = lng2 - lng1 #
    dlat = lat2 - lat1  #
    deg2grad = np.pi/180 #
    cos_dlat = torch.cos(deg2grad*dlat/2)
    term0 = dlng*cos_dlat
    term1 = term0*term0
    term2 = dlat*dlat
    loss_vect = torch.sqrt(term1 + term2)
    loss = 6371*torch.mean(loss_vect)
    return loss

def my_mse(Y1, Y2) :
    lng1, lat1 = Y1[:,0], Y1[:,1] #
    lng2, lat2 = Y2[:,0], Y2[:,1] #
    dlng = lng2 - lng1 #
    dlat = lat2 - lat1  #
    term1 = dlng**2
    term2 = dlat**2
    loss_vect = torch.sqrt(relu(term1 + term2))
    loss = 6371*torch.mean(loss_vect)
    return loss




