
#---------------------------------------------------------------------
## Import libraries
#----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm 
tqdm.pandas()

#---------------------------------------------------------------------
## Load arrival points of all the trajectories
#----------------------------------------------------------------------
def get_arrival(s) :
    """
    Get arrival point from string POLYLINE
    """
    arr_str = '[' + s.split('[')[-1].split(']')[0] + ']'
    return eval(arr_str)

arrivals = pd.read_csv("../CleanData/train_clean.csv", usecols=["POLYLINE"])
arrivals = arrivals["POLYLINE"].progress_apply(lambda x : get_arrival(x))
arrivals_np = np.vstack(arrivals)
print("Number of points: ", arrivals.shape[0])


#---------------------------------------------------------------------
## KMeans Clustering
#----------------------------------------------------------------------
# Run the k-means clustering
num_clusters = 1500
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(arrivals_np)

# Save the cluster centers and the data labels into variables
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

print("Number of clusters: ", len(cluster_centers))

# Save clusters centers
df_centers = pd.DataFrame(data=cluster_centers, columns=['x', 'y'])
df_centers.to_csv("cluster_centers.csv")

# Save points' labels 
arrivals = np.concatenate((arrivals_np, labels[:,None]), axis=1)
arrivals = pd.DataFrame(data=arrivals, columns=['lon', 'lat', 'cluster'])
arrivals.to_csv("arrivals_clustering.csv", index=False)



#---------------------------------------------------------------------
## Visualization
#----------------------------------------------------------------------
figure(figsize=(10,7))
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', color='r', s=5, alpha=.5)
plt.title("Cluster Centers")
plt.show()
