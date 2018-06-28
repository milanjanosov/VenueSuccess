import numpy as np
import matplotlib.pyplot as plt
import mpu
import os
import random
import sys
import time
import itertools
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from collections import Counter




'''  ---------------------------------------------------------  '''
'''   parse the coordinates into different strucutres           '''
'''  ---------------------------------------------------------  '''

def get_users_coordinates_ba(city, outfolder):

    users_coordinates = {}

    for line in open(outfolder + '/user_info/' + city + '_user_coordinates_raw_locals_filtered.dat'):
        if 'userid' not in line:
            fields = line.strip().split('\t')
            if len(fields[1:]) > 0:
                users_coordinates[fields[0]] = list(zip(*[tuple([float(fff.split(',')[0]), float(fff.split(',')[1])]) for fff in fields[1:]]))
                    
    return users_coordinates



def get_users_coordinates_db(city, outfolder):

    users_coordinates = {}

    for line in open(outfolder + '/user_info/' + city + '_user_coordinates_raw_locals_filtered.dat'):
        if 'userid' not in line:
            fields = line.strip().split('\t')
            if len(fields[1:]) > 0:
                users_coordinates[fields[0]] = [tuple([float(fff.split(',')[0]), float(fff.split(',')[1])]) for fff in fields[1:]]
                    
    return users_coordinates




'''  ---------------------------------------------------------  '''
'''     calc the centroids                                      '''
'''  ---------------------------------------------------------  '''

def get_centroids(c):

     return (sum(c[0])/len(c[0]),sum(c[1])/len(c[1]))



### calc the avg distance of points from eachother
def get_points_avg_dist(users_coordinates): 
    
    points_distance = {}
    # c = [(x1, x2, ...), (y1, y2, ...)]

    for u, c in users_coordinates.items():

        for c1 in (list(zip(*c))):

            c1str = '_'.join(list([str(fff) for fff in c1]))

            for c2 in (list(zip(*c))):
                if c1 != c2:
                    distance = mpu.haversine_distance((c1[1], c1[0]), (c2[1], c2[0]))

                    if c1str not in points_distance:
                        points_distance[c1str] = distance
                    else:
                        points_distance[c1str] += distance

            if c1str in points_distance:        
                points_distance[c1str] = points_distance[c1str]/len(c[1])
            else:
                points_distance[c1str] = 0.0

    return points_distance



'''  ---------------------------------------------------------  '''
'''   calc users centroids as they are                          '''
'''  ---------------------------------------------------------  '''

def get_users_centroids(city, outfolder, sample, LIMIT_num = 0, plot = True):

    users_coordinates = get_users_coordinates_ba(city, outfolder)    
    user_centroids    = {}
    users_numc        = []    
    users             = list(users_coordinates.keys())
    f, ax             = plt.subplots(4, 4, figsize=(20, 15))


    print('Get basic centroids...')


    if sample:
        user_sample     = sorted(random.sample([uu for uu in users if len(users_coordinates[uu][0]) == LIMIT_num], 16))
        indicies        = [(i,j) for i in range(4) for j in range(4)]
        user_sample_ind = [(user_sample[k], indicies[k][0], indicies[k][1]) for  k in range(len(user_sample))]
    else:
        user_sample     = [uu for uu in users if len(users_coordinates[uu][0]) == LIMIT_num]
        user_sample_ind = [(user_sample[k], 0, 0) for  k in range(len(user_sample))]


    for u, aa, bb in user_sample_ind:

        c = users_coordinates[u]
        users_numc.append(len(c[0]))
        
        centr = get_centroids(c)
        user_centroids[u] = centr

    if sample: 
        for (u, i, j) in user_sample_ind:
            ax[i,j].plot(users_coordinates[u][0], users_coordinates[u][1], 'bo', alpha = 0.35, markersize = 12, label = u)          
            ax[i,j].plot(user_centroids[u][0],    user_centroids[u][1],    'ro', markersize = 8)
            ax[i,j].legend(loc = 'left', fontsize = 9)

            
    if sample: plt.legend()
    if sample: plt.suptitle('CENTROIDS - 25 random users with > ' + str(LIMIT_num) + ' venues')
    if sample: plt.savefig(outfolder + 'figures/user_homes/' + city + '_centroids_example_' + str(LIMIT_num) + '.png')


  
    if plot: plt.show()
    else:    plt.close()    
  


    f = open(outfolder + '/user_homes/centroids/' + city + '_user_homes_centroids_' + str(LIMIT_num) + '.dat', 'w')
    for user, centr in user_centroids.items():
        f.write(user + '\t' + str(centr[0]) + '\t' + str(centr[1]) + '\n')
    f.close()


    if sample: plt.yscale('log')
    if sample: plt.hist(users_numc, bins = 100)
    if sample: plt.savefig(outfolder + '/figures/' + city + '_users_number_of_locations_' + str(LIMIT_num) + '.png')
    if sample: plt.close()  
    

    return user_sample_ind



'''  ---------------------------------------------------------  '''
'''   get the centroids with the distance cutoff                '''
'''  ---------------------------------------------------------  '''

def get_users_centroids_with_cutoff(user_sample, city, outfolder, sample, LIMIT_num = 0, limit = 2.0, plot = True):


    print('Get centroids with cutoff ' + str(limit) + ' km and limit_n = ' + str(LIMIT_num) + ' ...')

    users_coordinates = get_users_coordinates_ba(city, outfolder)    
    points_distance   = get_points_avg_dist(users_coordinates)    
    user_centroids    = {}
    users             = list(users_coordinates.keys())
    f, ax             = plt.subplots(4, 4, figsize=(20, 15))

 
    for u, aa, bb in user_sample:

        c      = users_coordinates[u]
        clist2 = list(zip(*[ c1 for c1 in list(zip(*c)) if points_distance['_'.join(list([str(fff) for fff in c1]))] < limit  ]))   

        if len(clist2) > 0:
            centr = get_centroids(clist2)
            user_centroids[u] = centr
        else:
            centr = get_centroids(c)
            user_centroids[u] = centr
            


    if sample: 
        for (u, i, j) in user_sample:  
            if u in user_centroids:
                ax[i,j].plot(users_coordinates[u][0], users_coordinates[u][1], 'go', alpha = 0.30, markersize = 12, label = u)          
                ax[i,j].plot(user_centroids[u][0],    user_centroids[u][1],    'ro', markersize = 8)
                ax[i,j].legend(loc = 'left', fontsize = 9)

            
    
    if sample: plt.legend()
    if sample: plt.suptitle('CENTROIDS + CUTOFF = ' + str(limit) + 'km, - 25 random users with > _limit_ venues')    
    if sample: plt.savefig(outfolder + 'figures/user_homes/' + city + '_centroids_cutoff_' + str(limit) + 'km_example_' + str(LIMIT_num) + '.png')

    if plot: plt.show()
    else:    plt.close()    
  
     
    f = open(outfolder + '/user_homes/centroids/' + city + '_user_homes_centroids_cutoff=' + str(limit) + 'km_' + str(LIMIT_num) + '.dat', 'w')
    for user, centr in user_centroids.items():
        f.write(user + '\t' + str(centr[0]) + '\t' + str(centr[1]) + '\n')
    f.close()
     

    



'''  ---------------------------------------------------------  '''
'''         do the DBscan   '''
'''  ---------------------------------------------------------  '''

def doDBSCAN(X, ax, sample, eps, mins, user):

    centers = [[1, 1], [-1, -1], [1, -1]]

    #X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps, min_samples=mins).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    
    n_clusters_    = len(set(labels)) - (1 if -1 in labels else 0)
    n_labels       = Counter(labels)
    n_labelss      = Counter([lll for lll in labels if lll > -1])

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)  for each in np.linspace(0, 1, len(unique_labels))]
    centr  = []    


    if len(n_labelss) > 1:

        biggestcluster = n_labelss.most_common(1)[0][0]

        for k, col in zip(unique_labels, colors):

            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]

            if len(xy) > 0:    
             
                if k == -1:
                    col = [0, 0, 0, 1]

                if sample: ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),  markeredgecolor='k', markersize=12, alpha = 0.3)
                xy = X[class_member_mask & ~core_samples_mask]
                if sample: ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),  markeredgecolor='b', markersize=6, alpha = 0.3)
           
                if k == biggestcluster:
                    centr = get_centroids(list(zip(*[(cc[0], cc[1]) for cc in  list(X[class_member_mask])])))
                    if sample: ax.legend(loc = 'left', fontsize = 8)

    
    return centr
        






def get_db_centroids(user_sample, city, outfolder, sample, LIMIT_num = 0, eps = 0.02, mins = 3):
      

    fout              = open(outfolder + '/user_homes/centroids/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '.dat', 'w')
    users_coordinates = get_users_coordinates_db(city, outfolder)    
    if sample: f, ax  = plt.subplots(4, 4, figsize=(20, 15))
    else: ax = np.asarray([[0],[0],[0]])

    print('Start doing DBSCan - eps = ' + str(eps) + ' ...')

    for (user, i, j) in user_sample:
    
        c     = users_coordinates[user]  
        x     = np.asarray(users_coordinates[user])  
        centr = doDBSCAN(x, ax[i,j], sample, eps, mins, user)  

        if len(centr) == 0:
            centr = get_centroids( list(zip(*c)) )   
            
        fout.write(user + '\t' + str(centr[0]) + '\t' + str(centr[1]) + '\n')

        if sample: 
            ax[i,j].plot(centr[0], centr[1], 'ro', label = user)
            ax[i,j].legend(loc = 'left', fontsize = 9)



    if sample: plt.legend()
    if sample: plt.suptitle('DBSCAN - 25 random users with > _limit_ venues')
    if sample: plt.savefig(outfolder + 'figures/user_homes/' + city + '_dbscan_example_' + str(LIMIT_num) + '_' + str(eps) + '.png')


    plt.close()
    fout.close()

    






