# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nmf

def test1():  
    """ trouver minimum de nb itérations nécessaire """
    # plotter erreur sur données de test et d'apprentissage
    ratings = np.genfromtxt('data/data1/ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    eps = 5e-3 #10e-3,5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 10000   # nb_iter = 20 000  est suffisant selon le graphe 
    nb_norm = 100
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    plt.figure()
    plt.plot(iter_histo,test_error_histo)
    plt.show()
#    Nu_embedded = TSNE(n_components=2).fit_transform(Nu)
#    plt.figure()
#    plt.plot(Nu_embedded[:,0],Nu_embedded[:,1], 'b*')
    
def test2():
    """ trouver la meilleure couple de eps et eps_ui """
    ratings = np.genfromtxt('data/data1/ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    lb = 1e-5
    up = 1e-1
    nb_eps = 2
    eps,eps_ui,error_train,error_test = nmf.optim_eps(data_train,data_test,Nu,Ni,lb,up,nb_eps)
    np.save("result/data1/eps.npy",eps)
    np.save("result/data1/error_train.npy",error_train)
    np.save("result/data1/error_test",error_test)
    error_min = error_test.min()
    print("erreur minimale = ", error_min)
    f = plt.figure()
    plt.imshow(error_test,extent=[lb,up,lb,up])
    plt.colorbar()
    plt.title("error_test selon eps")
    plt.xlabel("epsilon de SGD")
    plt.ylabel("epsilon utilisateur idem")
    f.savefig("result/data1/erreur_eps.pdf")


""" nouvelle données avec plus d'information sur les utilisateurs"""
""" UserIDs range between 1 and 6040; MovieIDs range between 1 and 3952 """
def test3():
    """ trouver minimum de nb itérations nécessaire """
    """ 300 000 iter """
    ratings = np.load('data/data2/ratings.npy') 
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 3
    eps = 5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 300000
    nb_norm = 200
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    np.save("result/data2/iter_histo.npy",iter_histo)
    np.save("result/data2/test_error_histo.npy",test_error_histo)
    np.save("result/data2/train_error_histo.npy",train_error_histo)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/error_percent.pdf")

def test4():
    """ trouver meilleure eps """
    ratings = np.load('data/data2/ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 3   
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    lb = 1e-5
    up = 1e-1
    nb_eps = 50
    nb_iter = 300000
    eps,eps_ui,error_train,error_test = nmf.optim_eps(data_train,data_test,Nu,Ni,lb,up,nb_eps,nb_iter = nb_iter)
    np.save("result/data2/eps.npy",eps)
    np.save("result/data2/error_train.npy",error_train)
    np.save("result/data2/error_test.npy",error_test)
    error_min = error_test.min()
    print("erreur minimale = ", error_min)
    plt.figure()
    plt.imshow(error_test,extent=[lb,up,lb,up])
    plt.colorbar()
    plt.title("error_test selon eps")
    plt.xlabel("epsilon de SGD")
    plt.ylabel("epsilon utilisateur idem")
    plt.savefig("result/data2/error_eps.pdf")

error = np.load("")