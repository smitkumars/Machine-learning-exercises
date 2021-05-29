#!/usr/bin/env python
# coding: utf-8

# In[1]:


# K-Nearest Neighbors is an algorithm for supervised learning.
#Where the data is 'trained' with data points corresponding to their classification. 
#Once a point is to be predicted, 
#it takes into account the 'K' nearest points to it to determine it's classification.


# In[3]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('teleCust1000t.csv')
df.head(5)


# In[6]:


##Data Visualization and Analysis


# In[6]:


df['custcat'].value_counts()

##281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers


# In[7]:


df.hist(column='income', bins=50)


# In[11]:


#Feature set
#Lets define feature sets, X:


# In[12]:


df.columns


# In[9]:


#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[8]:


#What are our labels?

y = df['custcat'].values
y[0:5]


# In[10]:


## Normalize DATA

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[ ]:


Train Test Split
Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of being over-fit.

It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split. Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[12]:


###Classification
##K nearest neighbor (KNN)

from sklearn.neighbors import KNeighborsClassifier


# In[13]:


##Training
##Lets start the algorithm with k=4 for now:

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[14]:


#Predicting
##we can use the model to predict the test set:

yhat = neigh.predict(X_test)
yhat[0:5]


# In[15]:


##Accuracy evaluation

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[16]:


##Lets start the algorithm with k=4 for now:

k = 6
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[17]:


#Predicting
##we can use the model to predict the test set:

yhat = neigh.predict(X_test)
yhat[0:5]


# In[18]:


##Accuracy evaluation

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[19]:


##We can calculate the accuracy of KNN for different Ks.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[20]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[21]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




