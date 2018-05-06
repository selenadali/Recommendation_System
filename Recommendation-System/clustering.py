#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:45:27 2018

@author: astricot
"""
import numpy as np
from sklearn.cluster import KMeans

from sklearn import cluster,datasets
from sklearn.manifold import TSNE

def tSNE_Nu(file1,file2):
    N = np.load(file1)
    N_embed = TSNE(n_components=2, perplexity= 10).fit_transform(N)
    np.save(file2, N_embed)

def tSNE_Ni(file1,file2):
    N = np.load(file1)
    N_embed = TSNE(n_components=2).fit_transform(N.transpose())
    np.save(file2, N_embed)
    
def clustering(N_embed,nb):
    c = cluster.AgglomerativeClustering(nb)
    c.fit(N_embed)
    c_class = c.labels_

#    kmeans = KMeans(n_clusters=nb, random_state=0).fit(N_embed)
    
    return c_class
#
#
##    f1 = 'data/data2/Nu.npy'
#f1 = 'data/data2/dim_90/Nu.npy'
##    f2 = 'data/data2/Ni.npy'   
#f2 = 'data/data2/dim_90/Ni.npy'
##    f3 = 'data/data2/Nu_embed.npy'
##    f4 = 'data/data2/Ni_embed.npy' 
#f3 = 'data/data2/dim_90/Nu_embed.npy'
#f4 = 'data/data2/dim_90/Ni_embed.npy'
#tSNE_Nu(f1,f3)
#tSNE_Ni(f2,f4)
#
#f1 = 'data/data2/dim_90/Nu_embed.npy'
#Nu_embed = np.load(f1)
#nb_cluster = 20
#class_Nu = clustering(Nu_embed,nb_cluster)
##plt.figure()
##plt.scatter(Nu_embed[:,0], Nu_embed[:,1],c=class_Nu, marker='s', s=20)
##plt.show()
