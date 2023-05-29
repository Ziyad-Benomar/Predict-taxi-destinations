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

# Create all tensors on GPU as default
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Trace nan values
torch.autograd.set_detect_anomaly(True)

# Import the neural net and loss function classes
from neural_net import *




#---------------------------------------------------------------------
## Importing Clusters
#---------------------------------------------------------------------
cluster_centers = pd.read_csv("Clustering/cluster_centers.csv")
cluster_centers = np.array([[cluster_centers.iloc[i].x, cluster_centers.iloc[i].y] for i in range(cluster_centers.shape[0])])
cluster_centers = torch.tensor(cluster_centers).float().cuda()




#---------------------------------------------------------------------
## Creating the Neural Network
#---------------------------------------------------------------------
emb_dims = [10]*6
k = 5
hidden_layer_size = 300

net = MyNet(cluster_centers, emb_dims, k, hidden_layer_size=hidden_layer_size).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 

print("The number of parameters in the network : ",sum(p.numel() for p in net.parameters()))



#---------------------------------------------------------------------
## Training the Neural Network
#---------------------------------------------------------------------
# Train the neural net

criterion = my_mse
#criterion = loss_equirec

num_epochs = 100

# DO NOT CHANGE train_size AND block_size
# train_size is the real size of the dataset, block_size is the size of each scv file in TrainData
train_size = 11764351
block_size = 32000
batch_size = 1000

num_blocks = train_size//block_size # What is left will be used as validation set
num_batchs = block_size//batch_size

print("Number of blocks per epoch :", num_blocks)
print("Number of batchs per block :", num_batchs)

loss_arr_block = []
loss_arr_epoch = []

for epoch in range(num_epochs) :
    # Shuffle data
    if epoch > 0 :
        shuff_time = shuffle_data(num_blocks, block_size)
        print(f"\n--------- Shuffle data... {shuff_time/60} minutes")
    
    print(f"\nEpoch {epoch} ------------------------------------------------")
    Ti = time.time()
    epoch_loss = 0
    for block_ix in range(num_blocks) :
        data_block = load_datablock(block_ix)
        block_loss = 0
        for batch_ix in range(num_batchs) :
            # Get the training data
            meta, traj, dest = get_minibatch(data_block, batch_ix, batch_size) 
            # Forward function 
            dest_pred = net.forward(meta, traj)
            # Compute the loss
            loss = criterion(dest, dest_pred)
            # Backward function
            optimizer.zero_grad()
            loss.backward()
            # Clipping the gradients and updating the weights 
            # (clipping prevents from having very large values that bias the model)
            clipping_value = 5
            params = net.parameters()
            torch.nn.utils.clip_grad_value_(params, clipping_value)
            optimizer.step()

            # Compute total batch_loss
            #block_loss += loss.item()
            block_loss += loss.item()

        block_loss /= num_batchs
        loss_arr_block.append(block_loss)  

        # Compute total epoch loss
        epoch_loss += block_loss

        if block_ix%50==0 and block_ix>0: 
            print(f"   block nb : {block_ix} \t; loss = {np.array(loss_arr_block[-50:]).mean()}")

    epoch_loss /= num_blocks
    loss_arr_epoch.append(epoch_loss)
    print(f"--------- Epoch Loss = {epoch_loss}")
    Tf = time.time()
    print(f"--------- Epoch duration = {(Tf-Ti)/60} minutes")
    print(f"--------- Block training = {(Tf-Ti)/num_blocks} seconds")
    print(f"Loss array: {loss_arr_epoch}")
