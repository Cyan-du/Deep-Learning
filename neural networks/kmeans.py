
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd


# In[ ]:

def kmeans(dat1,clusters,iterations):
#assigning random points as centroid of initial cluster
    index=np.random.permutation(dat1.index)[:clusters]
    cluster=list()
    label=list()
    sse=[]
    d=dict()
    for i in index:
        cluster.append(np.array(dat1.ix[i,:]))
#calculating the distance of each datapoint from the clusters and assigning label to minimum distance
    for i in range(len(dat1)):
        dist=list()
        for j in range(len(cluster)):
            dist.append(np.sum((np.array(dat1.ix[i])-cluster[j])**2))
        label.append(np.argmin(dist))
    dat1['label']=np.array(label)
    for i in range(iterations):
#calculating mean of clusters as new centroid
        cluster=list()
        label=list()
        for i in np.unique(dat1['label']):
            cluster.append(np.array(np.mean(dat1.ix[dat1['label']==i,:-1])))
#calculating the distance of each datapoint from the clusters and assigning label to minimum distance
        for i in range(len(dat1)):
            dist=list()
            for j in range(len(cluster)):
                dist.append(np.sum((np.array(dat1.ix[i,:-1])-cluster[j])**2))
            label.append(np.argmin(dist))
        dat1['label']=np.array(label)
#calculating the sum of squared error of individual cluster
    for i in range(len(cluster)):
        sse.append(np.sum((np.array(dat1.ix[dat1['label']==i,:-1])-cluster[i])**2))
#calculating the total sum of squared error
    totalsse=np.array(sse).sum()
    d['clustermean']=cluster
    d['data']=dat1
    d['sse']=sse
    d['totalsse']=totalsse
    return d

