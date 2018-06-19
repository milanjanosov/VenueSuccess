import os
import mpu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 



'''  ----------------------------------------------------------  '''
'''         parse the users coordinates from var inputs          '''
'''  ----------------------------------------------------------  '''

def get_users_homes(fn, city, outfolder, num = 3):
    
    users_coordinates = {}

    for line in open(outfolder + fn):
        if num == 4:
            user, lng, lat, venue = line.strip().split('\t')
        else:
            user, lng, lat = line.strip().split('\t')

        users_coordinates[user] = (float(lng), float(lat))
        
    return users_coordinates




'''  ----------------------------------------------------------  '''
'''         parse the users known home locations                 '''
'''  ----------------------------------------------------------  '''

def get_groundtruth_homes(city, outfolder):

    return get_users_homes( 'user_info/' + city + '_groundtruth_home_locations_unique.dat', city, outfolder, 4)




'''  ----------------------------------------------------------  '''
'''         read the assumed home locations by dif methods       '''
'''  ----------------------------------------------------------  '''

def get_homes_from_methods(city, outfolder, LIMIT_num):
    
    files = [ '/user_homes/centroids_filtered/' + fff for fff in os.listdir(outfolder + '/user_homes/centroids_filtered/' ) if '_' + str(LIMIT_num) + '_' in fff]

    methods_homes = {}
    for fn in files:

        method      = fn.split('homes_')[1].replace('_' + str(LIMIT_num) + '.dat' ,'').split('_filtered')[0]


        users_homes = get_users_homes(fn, city, outfolder)


      
        for user, home in users_homes.items():

            if user not in methods_homes:
                methods_homes[user] = {}

            methods_homes[user][method] = home
        

    #{user1: {method1 : predhome1, method2 :  predhome2, ...}, user2: {method2 : predhome3}, ... } ... }
    for k, v in methods_homes.items():
        print(k)

    return methods_homes




'''  ----------------------------------------------------------  '''
'''         calc the assumed homes from the groundtruth          '''
'''  ----------------------------------------------------------  '''

def get_distance_from_groundtruth(methods_homes, groundtruth_homes, city, outfolder):

    homedistances_users_methods = {}

    for user, home in groundtruth_homes.items():
        if user in methods_homes:
            for method, home_ in methods_homes[user].items():
                if user not in homedistances_users_methods:
                    homedistances_users_methods[user] = {}

                homedistances_users_methods[user][method] =  mpu.haversine_distance((home[1], home[0]), (home_[1], home_[0])) 

    return pd.DataFrame.from_dict(homedistances_users_methods, orient = 'index')


 
'''  ----------------------------------------------------------  '''
'''         viz the assumed homes dist from the groundtruth      '''
'''  ----------------------------------------------------------  '''

def get_final_comp_results(city, outfolder, LIMIT_num):

    groundtruth_homes         = get_groundtruth_homes(city, outfolder)   
    methods_homes             = get_homes_from_methods(city, outfolder, LIMIT_num)
    distance_from_groundtruth = get_distance_from_groundtruth(methods_homes, groundtruth_homes, city, outfolder)

    df_res = pd.DataFrame()
    df_res['Averages'] = distance_from_groundtruth.mean(axis=0)
    #df_res['Stdevs']   = distance_from_groundtruth.std(axis=0)

    df_res.to_csv(outfolder + '/user_homes/comparison/' + city + '_CENTROID_COMPARISON_RES_' + str(LIMIT_num) + '.csv', sep = ',', float_format='%.3f')





def plot_final_results(city, outfolder):

    files          = sorted([fff for fff in os.listdir(outfolder   + '/user_homes/comparison/') if 'csv' in fff])#+ city + '_CENTROID_COMPARISON_RES_' + str(LIMIT_num) + '.csv')
    f, ax          = plt.subplots(1, 2, figsize=(15, 7))
    methods_series = {}



    ''' put the centroid methods on the plot '''

    '''methods_series_centroid = {}
    methods_series_cutoff   = {}
    methods_series_dbscan   = {}

    for fn in files:

        index = int(fn.split('_')[-1].split('.')[0])


        for line in open(outfolder +  '/user_homes/comparison/' + fn):
            if 'Averages' not in line:


                #try:
                #method, avg, std =  line.strip().split(',')
                method, avg =  line.strip().split(',')
                avg = float(avg)
                #std = float(std)

                if 'dbscan' in method:

                    method, index = method.rsplit('_',  1)
                   # print(method, index)
                    if method not in methods_series_dbscan:
                        methods_series_dbscan[method] = [(float(index), avg)]
                    else:
                        methods_series_dbscan[method].append((float(index), avg))
                

                elif 'cutoff' in method:

                    a, method, index = method.rsplit('_')

                    if method not in methods_series_cutoff:
                        methods_series_cutoff[method] = [(float(index), avg)]
                    else:
                        methods_series_cutoff[method].append((float(index), avg))



                elif 'cutoff' not in method and 'centroid' in method:
            
                    method, index = method.split('_')

                    if method not in methods_series_centroid:
                        methods_series_centroid[method] = [(float(index), avg)]
                    else:
                        methods_series_centroid[method].append((float(index), avg))

                




    for m, s in methods_series_dbscan.items():
        s.sort(key=lambda tup: tup[0])     
        ind, avg = zip(*s)
        ax[0].plot(ind, avg, 'o-', label = m)


    for m, s in methods_series_cutoff.items():
        s.sort(key=lambda tup: tup[0])
        ind, avg = zip(*s)
        ax[0].plot(ind, avg, 'o-', label = m)


    for m, s in methods_series_centroid.items():
        s.sort(key=lambda tup: tup[0])
        ind, avg = zip(*s)
        ax[0].plot(ind, avg, 'o-', label = m)






    ### THIS WILL BE SPARES BECAUSE THERE ARE MULTIPLE CONDITIONS
    ###    - HAS UNIQUE GROUNDTRUTH LOCATION (50 PPL...)
    ###    - LOCAL OR UNKNOWN USER
    ###    - HAVE CAREER LONGER THAN LIMIT
    ###    - GET A HOME LOCATION WITHIN THE CITY
    ax[0].legend(loc = 'left', fontsize = 8)    
    ax[1].legend(loc = 'left', fontsize = 8)    

    plt.show()
    '''

    '''ax[0].set_xlabel('Min. number of locations/user')
    ax[0].set_ylabel('Avg distance between estimated and real residence location [km]')
    ax[1].set_xlabel('Min. number of locations/user')
    ax[1].set_ylabel('Avg distance between estimated and real residence location [km]')
    ax[0].legend(loc = 'left', fontsize = 8)
    ax[1].legend(loc = 'left', fontsize = 8)
    plt.savefig(outfolder   + '/figures/user_homes/' + city + '_home_distance_from_groundtruth_FULL.png')
    plt.show() 
    '''




    ''' compare the ML homes with the best centroid method '''

    best_method = 'dbscan_0.02_3'
    ML_folder   = outfolder + '/user_homes/MLhomes_filtered/'
    ML_files    = os.listdir(ML_folder)



    users_classifiers = {}
    groundtruth_homes = get_groundtruth_homes(city, outfolder) 
    


    for ind, fn in enumerate(ML_files):
    
       # if ind == 10: break


        LIMIT = int(fn.split('predicted_homes_')[1].split('_pred_home')[0])
        best_centroids = {} 
        for line in open(outfolder + '/user_homes/centroids_filtered/' + city + '_user_homes_'  + best_method + '_' + str(LIMIT) + '_filtered.dat' ):
            user, lng, lat = line.strip().split('\t')
            best_centroids[user] = (lng, lat)






        classifier = fn.split('pred_home_')[1].replace('.dat', '')

        for line in open(ML_folder + fn):
            user, lng1, lat1 = line.strip().split('\t')

            if user in best_centroids.keys():
                lng2, lat2 = best_centroids[user]
                dist = mpu.haversine_distance((float(lat1), float(lng1)), (float(lat2), float(lng2)))
    
                if classifier not in users_classifiers:
                    users_classifiers[classifier] = {}
                
                if LIMIT not in users_classifiers[classifier]:
                    users_classifiers[classifier][LIMIT] = [dist]
                else:
                    users_classifiers[classifier][LIMIT].append(dist)
    


    users_classifiers_avg = {}

    for c, indist in users_classifiers.items():
        
        if c not in users_classifiers_avg:
            users_classifiers_avg[c] = []

        for ind, dist in indist.items():

            users_classifiers_avg[c].append((ind, np.mean(dist))) 

            #print(c, ind, np.mean(dist))
   

    for c, data in users_classifiers_avg.items():

        ind, dist = zip(*data)

        ax[1].plot(ind, dist, 'o-', label = c)

    '''for m, s in users_classifiers.items():
        s.sort(key=lambda tup: tup[0])
        ind, dist = zip(*s)
        print(ind)
        
    '''
    plt.show()









