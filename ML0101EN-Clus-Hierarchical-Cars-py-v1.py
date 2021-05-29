#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width = 400, align = "center"></a>
# 
# # <center>Hierarchical Clustering</center>

# Welcome to Lab of Hierarchical Clustering with Python using Scipy and Scikit-learn package.

# #  Hierarchical Clustering - Agglomerative
# 
# We will be looking at a clustering technique, which is <b>Agglomerative Hierarchical Clustering</b>. Remember that agglomerative is the bottom up approach. <br> <br>
# In this lab, we will be looking at Agglomerative clustering, which is more popular than Divisive clustering. <br> <br>
# We will also be using Complete Linkage as the Linkage Criteria. <br>
# <b> <i> NOTE: You can also try using Average Linkage wherever Complete Linkage would be used to see the difference! </i> </b>

# In[1]:


import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# ### Generating Random Data
# We will be generating a set of data using the <b>make_blobs</b> class. <br> <br>
# Input these parameters into make_blobs:
# <ul>
#     <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
#     <ul> <li> Choose a number from 10-1500 </li> </ul>
#     <li> <b>centers</b>: The number of centers to generate, or the fixed center locations. </li>
#     <ul> <li> Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]]) </li> </ul>
#     <li> <b>cluster_std</b>: The standard deviation of the clusters. The larger the number, the further apart the clusters</li>
#     <ul> <li> Choose a number between 0.5-1.5 </li> </ul>
# </ul> <br>
# Save the result to <b>X1</b> and <b>y1</b>.

# In[2]:


X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)


# Plot the scatter plot of the randomly generated data

# In[3]:


plt.scatter(X1[:, 0], X1[:, 1], marker='o') 


# ---
# ### Agglomerative Clustering
# We will start by clustering the random data points we just created.

# The <b> Agglomerative Clustering </b> class will require two inputs:
# <ul>
#     <li> <b>n_clusters</b>: The number of clusters to form as well as the number of centroids to generate. </li>
#     <ul> <li> Value will be: 4 </li> </ul>
#     <li> <b>linkage</b>: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. </li>
#     <ul> 
#         <li> Value will be: 'complete' </li> 
#         <li> <b>Note</b>: It is recommended you try everything with 'average' as well </li>
#     </ul>
# </ul> <br>
# Save the result to a variable called <b> agglom </b>

# In[4]:


agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')


# Fit the model with <b> X2 </b> and <b> y2 </b> from the generated data above.

# In[5]:


agglom.fit(X1,y1)


# Run the following code to show the clustering! <br>
# Remember to read the code and comments to gain more understanding on how the plotting works.

# In[6]:


# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')



# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()


# 
# ### Dendrogram Associated for the Agglomerative Hierarchical Clustering
# Remember that a <b>distance matrix</b> contains the <b> distance from each point to every other point of a dataset </b>. <br>
# Use the function <b> distance_matrix, </b> which requires <b>two inputs</b>. Use the Feature Matrix, <b> X2 </b> as both inputs and save the distance matrix to a variable called <b> dist_matrix </b> <br> <br>
# Remember that the distance values are symmetric, with a diagonal of 0's. This is one way of making sure your matrix is correct. <br> (print out dist_matrix to make sure it's correct)

# In[7]:


dist_matrix = distance_matrix(X1,X1) 
print(dist_matrix)


# Using the <b> linkage </b> class from hierarchy, pass in the parameters:
# <ul>
#     <li> The distance matrix </li>
#     <li> 'complete' for complete linkage </li>
# </ul> <br>
# Save the result to a variable called <b> Z </b>

# In[8]:


Z = hierarchy.linkage(dist_matrix, 'complete')


# A Hierarchical clustering is typically visualized as a dendrogram as shown in the following cell. Each merge is represented by a horizontal line. The y-coordinate of the horizontal line is the similarity of the two clusters that were merged, where cities are viewed as singleton clusters. 
# By moving up from the bottom layer to the top node, a dendrogram allows us to reconstruct the history of merges that resulted in the depicted clustering. 
# 
# Next, we will save the dendrogram to a variable called <b>dendro</b>. In doing this, the dendrogram will also be displayed.
# Using the <b> dendrogram </b> class from hierarchy, pass in the parameter:
# <ul> <li> Z </li> </ul>

# In[9]:


dendro = hierarchy.dendrogram(Z)


# ## Practice
# We used __complete__ linkage for our case, change it to __average__ linkage to see how the dendogram changes.

# In[ ]:


# write your code here



# Double-click __here__ for the solution.
# 
# <!-- Your answer is below:
#     
# Z = hierarchy.linkage(dist_matrix, 'average')
# dendro = hierarchy.dendrogram(Z)
# 
# -->

# ---
# # Clustering on Vehicle dataset
# 
# Imagine that an automobile manufacturer has developed prototypes for a new vehicle. Before introducing the new model into its range, the manufacturer wants to determine which existing vehicles on the market are most like the prototypes--that is, how vehicles can be grouped, which group is the most similar with the model, and therefore which models they will be competing against.
# 
# Our objective here, is to use clustering methods, to find the most distinctive clusters of vehicles. It will summarize the existing vehicles and help manufacture to make decision about new models simply.

# ### Download data
# To download the data, we will use **`!wget`**. To download the data, we will use `!wget` to download it from IBM Object Storage.  
# __Did you know?__ When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)

# In[ ]:


get_ipython().system('wget -O cars_clus.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv')


# ## Read data
# lets read dataset to see what features the manufacturer has collected about the existing models.

# In[10]:


filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)

pdf.head(5)


# The featuresets include  price in thousands (price), engine size (engine_s), horsepower (horsepow), wheelbase (wheelbas), width (width), length (length), curb weight (curb_wgt), fuel capacity (fuel_cap) and fuel efficiency (mpg).

# ### Data Cleaning
# lets simply clear the dataset by dropping the rows that have null value:

# In[11]:


print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)


# ### Feature selection
# Lets select our feature set:

# In[12]:


featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]


# ### Normalization
# Now we can normalize the feature set. __MinMaxScaler__ transforms features by scaling each feature to a given range. It is by default (0, 1). That is, this estimator scales and translates each feature individually such that it is between zero and one.

# In[13]:


from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]


# ## Clustering using Scipy
# In this part we use Scipy package to cluster the dataset:  
# First, we calculate the distance matrix. 

# In[14]:


import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])


# In agglomerative clustering, at each iteration, the algorithm must update the distance matrix to reflect the distance of the newly formed cluster with the remaining clusters in the forest. 
# The following methods are supported in Scipy for calculating the distance between the newly formed cluster and each:
#     - single
#     - complete
#     - average
#     - weighted
#     - centroid
#     
#     
# We use __complete__ for our case, but feel free to change it to see how the results change.

# In[15]:


import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')


# Essentially, Hierarchical clustering does not require a pre-specified number of clusters. However, in some applications we want a partition of disjoint clusters just as in flat clustering.
# So you can use a cutting line:

# In[16]:


from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters


# Also, you can determine the number of clusters directly:

# In[17]:


from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters


# Now, plot the dendrogram:

# In[18]:


fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')


# ## Clustering using scikit-learn
# Lets redo it again, but this time using scikit-learn package:

# In[19]:


dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)


# Now, we can use the 'AgglomerativeClustering' function from scikit-learn library to cluster the dataset. The AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. The linkage criteria determines the metric used for the merge strategy:
# 
# - Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
# - Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# - Average linkage minimizes the average of the distances between all observations of pairs of clusters.

# In[20]:


agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_


# And, we can add a new field to our dataframe to show the cluster of each row:

# In[21]:


pdf['cluster_'] = agglom.labels_
pdf.head()


# In[22]:


import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')


# As you can see, we are seeing the distribution of each cluster using the scatter plot, but it is not very clear where is the centroid of each cluster. Moreover, there are 2 types of vehicles in our dataset, "truck" (value of 1 in the type column) and "car" (value of 1 in the type column). So, we use them to distinguish the classes, and summarize the cluster. First we count the number of cases in each group:

# In[23]:


pdf.groupby(['cluster_','type'])['cluster_'].count()


# Now we can look at the characterestics of each cluster:

# In[24]:


agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars


# 
# It is obvious that we have 3 main clusters with the majority of vehicles in those.
# 
# __Cars__:
# - Cluster 1: with almost high mpg, and low in horsepower.
# - Cluster 2: with good mpg and horsepower, but higher price than average.
# - Cluster 3: with low mpg, high horsepower, highest price.
#     
#     
#     
# __Trucks__:
# - Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
# - Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
# - Cluster 3: with good mpg and horsepower, low price.
# 
# 
# Please notice that we did not use __type__ , and __price__ of cars in the clustering process, but Hierarchical clustering could forge the clusters and discriminate them with quite high accuracy.

# In[25]:


plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')


# ## Want to learn more?
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: [SPSS Modeler](http://cocl.us/ML0101EN-SPSSModeler).
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at [Watson Studio](https://cocl.us/ML0101EN_DSX)
# 
# ### Thanks for completing this lesson!
# 
# Notebook created by: <a href = "https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>
# 
# <hr>
# Copyright &copy; 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).​

# In[ ]:




