import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
import mpu
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib as m

# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


'''  ---------------------------------------------------------  '''
'''                  do the DBscan                              '''
'''
    inputs like:
        X       : the coordinates of the user
        sample  : a bool, if sample = True than it creates a random sample of the user set and creates plots as well
                          if sample = False it processes the whole user set and doesn't create any plots
        eps, mins: DBscan params


'''
'''  ---------------------------------------------------------  '''

def doDBSCAN(X, sample, eps, mins):

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

                xy = X[class_member_mask & ~core_samples_mask]

        
                if k == biggestcluster:
                    centr = get_centroids(list(zip(*[(cc[0], cc[1]) for cc in  list(X[class_member_mask])])))


    
    return centr
        



'''  ------------------------------------------------------------------------------------  '''
'''     this calculates the centroid of a list of coordinates, which come in the form of   '''
'''
         c = [(x1, x2, x3, ...), (y1, y2, y3, ...)]
'''
'''  ------------------------------------------------------------------------------------  '''

def get_centroids(c):

     return (sum(c[0])/len(c[0]),sum(c[1])/len(c[1]))




'''  -------------------------------------------------------------------------------------------------------  '''
'''    this reads my awkward fileformat and creates the coordinate lists of users in dict users_coordinates   '''
'''     which looks like:
        users_coordinates[user] = [((x1, y1), (x2, y2), (x3, y3), ...]
'''
'''  -------------------------------------------------------------------------------------------------------  '''

def get_users_coordinates_db(city, outfolder):

    users_coordinates = {}

    # this iterates over the coordinate files, i added my example 
    for line in open(outfolder + '/user_info/' + city + '_user_coordinates_raw_locals_filtered.dat'):
        if 'userid' not in line:
            fields = line.strip().split('\t')
    
            if len(fields[1:]) > 0:
                users_coordinates[fields[0]] = [tuple([float(fff.split(',')[0]), float(fff.split(',')[1])]) for fff in fields[1:]]
                    
    return users_coordinates





'''  -------------------------------------------------------------------------------------------------------------  '''
'''    runs the dbscan algo on either sample users (16 random users, creating also plot), or doing the whole stuff  '''
'''     
        city, outfoldr: these are just needed for the input/output folders, rewrite it the way you prefer
        sample        : true/false whether you want to have plots about 16 users or everything on everyone
        LIMIT_num     : control parameter, considering only users having more locations than this number
        eps, mins     : the DBscan params ( http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

'''
'''  -------------------------------------------------------------------------------------------------------  '''

def get_db_centroids(city, outfolder, sample, LIMIT_num = 0, eps = 0.02, mins = 3):
      

    # just save the centroid coordinates somewhere
    outfile           = outfolder + '/user_homes/optimize_centroids/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '.dat'
    fout              = open(outfile, 'w')
    users_coordinates = get_users_coordinates_db(city, outfolder)    

   
    #if working with just a sample of 16 (to plot them), this is how i pick those
    if sample:
        user_sample = random.sample([uu for uu in users_coordinates.keys() if len(users_coordinates[uu]) > LIMIT_num], 16)
        indicies    = [(i,j) for i in range(4) for j in range(4)]
        users       = [(user_sample[k], indicies[k][0], indicies[k][1]) for  k in range(len(user_sample))]
    else:
        users       = [(u, 0, 0) for u in users_coordinates if len(users_coordinates[u]) > LIMIT_num]



    # get the users centroids 
    # centr contains those    
    for (user, i, j) in users:


    
        c     = users_coordinates[user]        
        x     = np.asarray(c)  
        centr = doDBSCAN(x, sample, eps, mins)  

        if len(centr) == 0:
            centr = get_centroids( list(zip(*c)) )   
            
        fout.write(user + '\t' + str(centr[0]) + '\t' + str(centr[1]) + '\n')


    fout.close()



    # if you are dealing with the samples, this just creates the ugly plot
    # you can just save it or whatever
    if sample: 
        plt.legend()
        plt.suptitle('DBSCAN - 25 random users with > _limit_ venues')
        #plt.savefig(outfolder + 'figures/user_homes/' + city + '_dbscan_example_' + str(LIMIT_num) + '_' + str(eps) + '.png')
        plt.show()
    

    return outfile


'''  ----------------------------------------------------------  '''
'''         parse the users known home locations                 '''
'''  ----------------------------------------------------------  '''
    
def get_users_homes(city, outfolder):
    

    fn = 'user_info/' + city + '_groundtruth_home_locations_unique.dat'
    users_coordinates = {}

    for line in open(outfolder + fn):
        user, lng, lat, venid   = line.strip().split('\t')
        users_coordinates[user] = (float(lng), float(lat))
        
    return users_coordinates




'''  ----------------------------------------------------------------------------------  '''
'''   get the distance between the groundtruth home locations and the dbscan results
        - as a function of eps, mins, and LIMIT

'''
'''  ----------------------------------------------------------------------------------  '''
    
def get_differences(fout, city, outroot, resfile, users_homes, LIMIT_, eps_, mins_):


    users_centroids = {}   
    dists           = []

    for line in open(resfile):
        user, lng, lat = line.strip().split()
        if user in users_homes:

            lng1 = float(lng)
            lat1 = float(lat)

            lng2 = users_homes[user][0]
            lat2 = users_homes[user][1]

            dists.append(mpu.haversine_distance((lat1, lng1), (lat2, lng2)))


    fout  = open(outroot + 'user_homes/optimize_centroids/series/dbscan_opt_limit_series_' + str(eps_) + '_' + str(mins_) + '.dat', 'a') 
    fout.write(str(LIMIT_) + '\t' + str( np.mean(dists)) + '\n')         
    fout.close()
    

    return users_centroids


'''  ----------------------------------------------------------------------------------  '''
'''   gopitmize the stuff and writing out files as
        (eps, mins) -> [(limit1, avgdist1), (limit2, avgdist2), ...]

'''
'''  ----------------------------------------------------------------------------------  '''
    


def optimize_db_scan(city, outroot):




    for eps_ in [0.001, 0.005, 0.01, 0.03, 0.1, 0.2]:

        for mins_ in [2,3,4,5, 6]:

            fout  = open(outroot + 'user_homes/optimize_centroids/series/dbscan_opt_limit_series_' + str(eps_) + '_' + str(mins_) + '.dat', 'w') 
            fout.close()

            for LIMIT_ in range(0, 15):

                print(eps_, '\t', mins_, '\t', LIMIT_)

                resfile     = get_db_centroids(city, outroot, sample = False, LIMIT_num = LIMIT_, eps = eps_, mins = mins_)

                users_homes = get_users_homes(city, outroot)

                get_differences(fout, city, outroot, resfile, users_homes, LIMIT_, eps_, mins_)





def visualize_opt_res(city, outroot):

    resfolder = outroot + 'user_homes/optimize_centroids/series/'
    files     = os.listdir(resfolder)
    f, ax     = plt.subplots(1, 2, figsize=(15, 5))


    x = []
    y = []
    z = []


    for fn in files:
        series = []
        for line in open(resfolder + fn):
            limit, dist = [float(fff) for fff in line.strip().split('\t')]
            series.append((limit, dist))
            
            if limit == 0:

                eps, mins = fn.rsplit('/', 1)[0].split('series_')[-1].replace('.dat', '').split('_')
                eps       = float(eps)      
                mins      = float(mins)

                x.append(float(eps))
                y.append(float(mins))
                z.append(dist)


        ax[0].plot([s[0] for s in series], [s[1] for s in series], label = fn.rsplit('/', 1)[0].split('series_')[-1].replace('.dat', '')  )

    ax[0].legend(loc = 'left', fontsize = 8)
    ax[0].set_xlabel('limit')
    ax[0].set_ylabel('avg dist between groundtruth and dbscan')

    im = ax[1].tripcolor(x,y,z)
    ax[1].plot(x,y, 'ko ')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('eps')
    ax[1].set_ylabel('mins')
    f.colorbar(im, ax=ax[1])

    plt.savefig(outroot   + '/figures/' + city + '_OPTIMIZE_DBscan.png')

    plt.show()

    

'''
city    = 'bristol'
outroot = '../ProcessedData/' + city + '/'




optimize_db_scan(city, outroot)
visualize_opt_res(city, outroot)

'''




