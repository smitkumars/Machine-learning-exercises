#!/usr/bin/env python
# coding: utf-8

# In[71]:


from pandas_datareader import data,wb

import pandas as pd
import seaborn as sns
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:





# In[72]:


start= datetime.datetime(2006,1,1)
end= datetime.datetime(2016,1,1)


# In[73]:


BAC= data.DataReader('BAC','yahoo',start,end)


# In[74]:


BAC= data.DataReader('BAC','yahoo',start,end)

C= data.DataReader('C','yahoo',start,end)

GS= data.DataReader('GS','yahoo',start,end)

JPM= data.DataReader('JPM','yahoo',start,end)

MS= data.DataReader('MS','yahoo',start,end)

WFC= data.DataReader('WFC','yahoo',start,end)


# In[75]:


BAC=BAC[["Open","High","Low","Close","Volume"]]

C=C[["Open","High","Low","Close","Volume"]]

GS=GS[["Open","High","Low","Close","Volume"]]

JPM=JPM[["Open","High","Low","Close","Volume"]]

MS=MS[["Open","High","Low","Close","Volume"]]

WFC=WFC[["Open","High","Low","Close","Volume"]]


# In[76]:


WFC


# In[77]:


BAC


# In[78]:


tickers=['BAC','C','GS','JPM','MS','WFC']


# In[79]:


tickers


# In[80]:


bank_stocks= pd.concat([BAC,C,GS,JPM,MS,WFC],axis=1,keys=tickers)


# In[81]:


bank_stocks


# In[82]:


bank_stocks.head()


# In[83]:


bank_stocks.columns.names=['Bank ticker','Stock info']


# In[84]:


bank_stocks.head()


# In[85]:


bank_stocks['BAC']['Close'].max()


# In[86]:


for tick in tickers:
    print(bank_stocks[tick]['Close'].max())


# In[87]:


bank_stocks.xs(key='Close',axis=1,level='Stock info')


# In[88]:


bank_stocks.xs(key='Close',axis=1,level='Stock info').max()


# In[89]:


returns=pd.DataFrame()


# In[90]:


for tick in tickers:
    returns[tick+' '+'Return']= bank_stocks[tick]['Close'].pct_change()


# In[91]:


returns.head()


# In[92]:


sns.pairplot(returns[1:])


# In[93]:


returns.head()


# In[94]:


returns.min()


# In[96]:


returns['BAC Return'].argmin()


# In[97]:


returns.idxmin()


# In[98]:


returns.idxmax()


# In[100]:


returns.std()


# In[102]:


returns.ix['2015-01-01':'2015-12-31'].std()


# In[105]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=50)


# In[106]:


sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='green',bins=50)


# In[107]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
import cufflinks as cf
cf.go_offline()


# In[108]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(label=tick,figsize=(12,4))


# In[109]:


bank_stocks.xs(key='Close',axis=1,level='Stock info').plot()


# In[111]:


bank_stocks.xs(key='Close',axis=1,level='Stock info').iplot()


# In[112]:


BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 day moving average')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC Close')
plt.legend()


# In[113]:



bank_stocks.xs(key='Close',axis=1,level='Stock info').corr()


# In[122]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock info').corr(),annot=True)


# In[121]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock info').corr(),annot=True)


# In[123]:


close_corr=  bank_stocks.xs(key='Close',axis=1,level='Stock info').corr()


# In[124]:


close_corr


# In[127]:


close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# In[128]:


bac15= BAC[['Open','High','Low','Close']].ix['2015-01-01':'2016-01-01']
bac15.iplot(kind='candle')


# In[129]:


MS['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,25])


# In[130]:


BAC['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')


# In[ ]:




