#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv('911.csv')
df.head()


# In[8]:


df.describe()


# In[10]:


df.info()


# In[15]:


df['zip'].value_counts().head(5)


# In[18]:


df['twp'].value_counts().head(5)


# In[19]:


df['title'].unique()


# In[20]:


len(df['title'].unique())


# In[21]:


x=df['title'].iloc[0]


# In[22]:


x


# In[24]:


x.split(':')


# In[25]:


x.split(':')[0]


# In[26]:


df['reason']=df['title'].apply(lambda title: title.split(':')[0])


# In[27]:


df['reason']


# In[28]:


df['reason'].value_counts()


# In[29]:


sns.countplot(x='reason',data=df)


# In[31]:


sns.countplot(x='reason',data=df,palette='viridis')


# In[34]:


type(df['timeStamp'].iloc[0])


# In[36]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# In[37]:


type(df['timeStamp'].iloc[0])


# In[38]:


time=df['timeStamp'].iloc[0]
time.hour


# In[39]:


time


# In[40]:


time.year


# In[41]:


time.dayofweek


# In[44]:


df['Hour']=df['timeStamp'].apply(lambda time: time.hour)


# In[45]:


df['Hour']


# In[46]:


df['Month']=df['timeStamp'].apply(lambda time: time.month)
df['Day of week']=df['timeStamp'].apply(lambda time: time.dayofweek)


# In[47]:


df.head()


# In[48]:


dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[49]:


df['Day of week']=df['Day of week'].map(dmap)


# In[50]:


df.head()


# In[51]:


sns.countplot(x='Day of week',data=df)


# In[52]:


sns.countplot(x='Day of week',data=df,hue='reason')


# In[55]:


sns.countplot(x='Day of week',data=df,hue='reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


# In[56]:


sns.countplot(x='Month',data=df,hue='reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


# In[59]:


bymonth = df.groupby('Month').count()


# In[60]:


bymonth.head()


# In[61]:


bymonth['lat'].plot()


# In[62]:


sns.countplot(x='Month',data=df,palette='viridis')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


# In[63]:


bymonth.reset_index()


# In[64]:


sns.lmplot(x='Month', y='twp',data=bymonth.reset_index())


# In[65]:


t=df['timeStamp'].iloc[0]


# In[68]:


df['Date']=df['timeStamp'].apply(lambda t:t.date())


# In[69]:


df.head()


# In[67]:


t.date()


# In[70]:


df.groupby('Date').count().head()


# In[72]:


df.groupby('Date').count()['lat'].plot()
plt.tight_layout()


# In[74]:


df[df['reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title('fire')
plt.tight_layout()


# In[75]:


df.groupby(by=['Day of week','Hour']).count()


# In[77]:


dayhour=df.groupby(by=['Day of week','Hour']).count()['reason'].unstack()


# In[78]:


sns.heatmap(dayhour)


# In[79]:


sns.heatmap(dayhour,cmap='viridis')


# In[81]:


plt.figure(figsize=(12,6))
sns.heatmap(dayhour,cmap='viridis')


# In[82]:


sns.clustermap(dayhour,cmap='viridis')


# In[84]:


daymonth=df.groupby(by=['Day of week','Month']).count()['reason'].unstack()
daymonth.head()


# In[85]:


sns.heatmap(daymonth,cmap='viridis')


# In[86]:


sns.clustermap(daymonth,cmap='viridis')


# In[ ]:




