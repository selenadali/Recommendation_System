#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:48:29 2018



"""
import clustering as cls
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_notebook, save


def write_tsne():
#    f1 = 'data/data2/Nu.npy'
    f1 = 'data/data2/dim_90/Nu.npy'
#    f2 = 'data/data2/Ni.npy'   
    f2 = 'data/data2/dim_90/Ni.npy'
#    f3 = 'data/data2/Nu_embed.npy'
#    f4 = 'data/data2/Ni_embed.npy' 
    f3 = 'data/data2/dim_90/Nu_embed.npy'
    f4 = 'data/data2/dim_90/Ni_embed.npy'
    cls.tSNE_Nu(f1,f3)
    cls.tSNE_Ni(f2,f4)

def movie_genre(list_genre, Ni_embed, movies):
    movies_index = []
    for i in range(len(list_genre)):
        movies_index.append(movies[movies['genres'].str.contains(list_genre[i])].index.values)
        Ni_emb_mov =  Ni_embed.take(movies_index[i],axis=0)
#        nom = "result/data2/movies_genre/" + list_genre[i] + ".pdf"
        nom = "result/data2/sub_movies_genre/" + list_genre[i] + ".pdf"
        plt.figure()
        plt.scatter(Ni_embed[:,0], Ni_embed[:,1],c=class_Ni, marker='s', s=20)
        plt.scatter(Ni_emb_mov[:,0], Ni_emb_mov[:,1],c='red', marker='o', s=10)
        plt.savefig(nom)
        plt.show()
        plt.close()
        
def user_gender(list_gender, Nu_embed, users, class_Nu, nb_cluster, count_total):
    users_gender = []
    tmp_class = np.zeros((len(list_gender), nb_cluster))

    for i in range(len(list_gender)):
        users_gender.append(users[users['Gender'] == list_gender[i] ].index.values)
        nom = 'tmp' + str(i)
        #tmp0 .... tmp6 categories de l'age
        nom = users_gender[i]
        for k in range(len(nom)):
            tmp_class[i][class_Nu[nom[k]]] += 1
            
        Nu_emb_user_gender =  Nu_embed.take(users_gender[i],axis=0)
#        nom = "result/data2/users_gender/" + list_gender[i] + ".pdf"
        nom = "result/data2/sub_users_gender/" + list_gender[i] + ".pdf"
        plt.figure()
        plt.scatter(Nu_embed[:,0], Nu_embed[:,1],c=class_Nu, marker='s', s=20)
        plt.scatter(Nu_emb_user_gender[:,0], Nu_emb_user_gender[:,1],c='red', marker='o', s=10)        
        plt.savefig(nom)
        plt.show()
        plt.close()     

    for c in range(nb_cluster):
        print("Cluster ", c) 
#        str_write += "Cluster " + str(c) + '\n'
#        print(tmp_class.transpose()[c][0], count_total[c])       
        print("Female :", (tmp_class.transpose()[c][0] / count_total[c]) * 100 , "%")
#        print(tmp_class.transpose()[c][1], count_total[c])
        print("Male :", (tmp_class.transpose()[c][1] / count_total[c]) * 100 , "%")
#            str_write += "Age group " + str(a) + ": "+ str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + "%" + '\n'
#            str_write += str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + '\n'

#        print(str_write)
#        file = open("result/data2/sub_users_age/age_group_5.txt", "w")
#        file.write(str_write) 




def user_age(list_age, Nu_embed, users, class_Nu, nb_cluster, count_total):
    users_age = []
    str_write = ""
    
    tmp_class = np.zeros((len(list_age), nb_cluster))
    liste_age_user = []
    for i in range(len(list_age)):
        users_age.append(users[(users['Age'] >= list_age[i][0]) & (users['Age'] < list_age[i][1]) ].index.values.tolist())

#        print(users_age)
        nom = 'tmp' + str(i)
        #tmp0 .... tmp6 categories de l'age
        nom = users_age[i]
        liste_age_user.append(nom)
        ###########################################################
        #############################################################
        print("******************", liste_age_user)
        ###########################################################
        #############################################################
        
        for k in range(len(nom)):
            tmp_class[i][class_Nu[nom[k]]] += 1
            
        Nu_emb_user_age =  Nu_embed.take(users_age[i],axis=0)
#        nom = "result/data2/users_age/" + str(list_age[i][0]) + ".pdf"
        nom = "result/data2/sub_users_age/" + str(list_age[i][0]) + ".pdf"
        plt.figure()
        plt.scatter(Nu_embed[:,0], Nu_embed[:,1],c=class_Nu, marker='s', s=20)
        plt.scatter(Nu_emb_user_age[:,0], Nu_emb_user_age[:,1],c='red', marker='o', s=10)        
        plt.savefig(nom)
        plt.show()
        plt.close() 
        
    for c in range(nb_cluster):
#        print("Cluster ", c) 
        str_write += "Cluster " + str(c) + '\n'
        for a in range(len(list_age)):
#            print("Age group", a, ":", (tmp_class.transpose()[c][a] / count_total[c]) * 100 , "%")
#            print(tmp_class.transpose()[c][a], count_total[c])
#            str_write += "Age group " + str(a) + ": "+ str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + "%" + '\n'
            str_write += str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + '\n'

#        print(str_write)
        file = open("result/data2/sub_users_age/age_group_5.txt", "w")
        file.write(str_write) 
        
 
        
def bokeh(users, Nu_embed, c_u):
    df_combine = users
    df_combine['x-tsne'] = Nu_embed[:,0]
    df_combine['y-tsne'] = Nu_embed[:,1]

    palette ={1:'red',2:'green',3:'blue',
              4:'yellow',5:"orange", 6:"pink",
              7:"black", 8:"purple", 9:"brown",0:"white"}
    colors =[]
    for i in c_u:
            colors.append(palette[i])

    print(colors)

    source = ColumnDataSource(dict(
        x=df_combine['x-tsne'],
        y=df_combine['y-tsne'],
        age= df_combine['Age'],
        gender= df_combine['Gender'],
        occupation = df_combine['Occupation'],
        color = colors
        ))

    hover_tsne = HoverTool( tooltips=[
                                     ("Age", "@age"),
                                     ("Gender", "@gender"),
                                     ("Occupation", "@occupation")
                                     ])
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']

    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='users')

    plot_tsne.circle('x', 'y', size=10, fill_color='color',
                     alpha=0.5, line_width=0, source=source, name="bokeh")

    # hover tools
    hover = plot_tsne.select(dict(type=HoverTool))
    hover.tooltips = {"content": "Age: @age, Gender: @gender, Occupation: @occupation"}

    show(plot_tsne)
    
def find_avg_age(nb_cluster, Nu_embed):
    for i in range(nb_cluster):
        avg = 0
        cl = np.where(class_Nu == i)
    #    nu_cl_zero = Nu_embed.take(cl_zero,axis=0)
        users_nb_cluster = users.iloc[cl]
        avg = users_nb_cluster['Age'].mean()
        median = users_nb_cluster['Age'].median()
        print("For cluster ",i, " age moyenne: ", avg )
        print("For cluster ",i, " age median: ", median )
   
def fing_gender_major(nb_cluster, Nu_embed):
    for i in range(nb_cluster):
        f = 0
        m = 0
        cl = np.where(class_Nu == i)
    #    nu_cl_zero = Nu_embed.take(cl_zero,axis=0)
        users_nb_cluster = users.iloc[cl]
        for j in users_nb_cluster["Gender"]:
            
            if j == "F":
                f += 1
            elif j == "M":
                m += 1
        print("For cluster ",i, " f: ", f, "m: ", m, "diff", f-m)
        
#def get_ratings_pd():
#    dtype = [('UserID','int'), ('MovieID','int'), ('Rating','int'), ('Timestamp','int')]
#    values = np.load("data/data2/ratings.npy")
#    values.astype(dtype)
#    index = [i for i in range(1, len(values)+1)]
#    ratings = pd.DataFrame(values, index=index)
#    ratings.rename(columns = {'f0':'UserID','f1':'MovieID','f2':'Rating','f3':'Timestamp'}, inplace=True)
#    return ratings

#f1 = 'data/data2/Nu_embed.npy'
f1 = 'data/data2/dim_90/Nu_embed.npy'
Nu_embed = np.load(f1)
nb_cluster = 20
class_Nu = cls.clustering(Nu_embed,nb_cluster)
#plt.figure()
#plt.scatter(Nu_embed[:,0], Nu_embed[:,1],c=class_Nu, marker='s', s=20)
#plt.show()

#f2 = 'data/data2/Ni_embed.npy'
f2 = 'data/data2/dim_90/Ni_embed.npy'
Ni_embed = np.load(f2)
class_Ni = cls.clustering(Ni_embed,nb_cluster)
#plt.figure()
#plt.scatter(Ni_embed[:,0], Ni_embed[:,1],c=class_Ni, marker='s', s=20)
#plt.show()


count_total = np.zeros(nb_cluster)
for j in range(nb_cluster):
    count_total[j] = np.count_nonzero(class_Nu == j)
    
    
movies = pd.read_csv("data/data2/sub_movies.csv", sep=",")
#list_genre = ["Action","Adventure","Animation",	"Children's",	"Comedy",	"Crime",	"Documentary",
#    "Drama",	"Fantasy",	"Film-Noir",	"Horror",	"Musical",	"Mystery",	"Romance",	"Sci-Fi",	"Thriller",
#	"War",	"Western"]
#
#movie_genre(list_genre, Ni_embed,movies)

##
#users = pd.read_table("data/data2/users.dat", sep="::", names= ['UserID','Gender','Age','Occupation','Zip-code'])
users = pd.read_csv("data/data2/sub_users.csv", sep=",")
#users_age = users[users['Age'] < 20 ].index.values

#bokeh(users, Nu_embed, class_Nu)
#
#find_avg_age(nb_cluster,Nu_embed)

list_gender = ["F", "M"]
#fing_gender_major(nb_cluster, Nu_embed)
#user_gender(list_gender, Nu_embed, users, class_Nu, nb_cluster,count_total)

##   *  1:  "Under 18"
##	* 18:  "18-24"
##	* 25:  "25-34"
##	* 35:  "35-44"
##	* 45:  "45-49"
##	* 50:  "50-55"
##	* 56:  "56+"

list_age = [[0,18],[18,24],[25,34],[35,44],[45,49],[50,55],[56,100]]
user_age(list_age, Nu_embed,users, class_Nu, nb_cluster,count_total)
