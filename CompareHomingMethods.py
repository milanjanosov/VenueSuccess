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
    
    files = ['/user_homes/centroids/' + fff for fff in os.listdir(outfolder + '/user_homes/centroids/' ) if '_user_homes_' in fff and '_' + str(LIMIT_num) + '.dat' in fff]



    methods_homes = {}
    for fn in files:



        method      = fn.split('homes_')[1].replace('_' + str(LIMIT_num) + '.dat' ,'')
        users_homes = get_users_homes(fn, city, outfolder)

        for user, home in users_homes.items():

            if user not in methods_homes:
                methods_homes[user] = {}

            methods_homes[user][method] = home
    
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

def dummy_viz_goodness_of_dist(distance_from_groundtruth, city, outfolder, LIMIT_num):

    def viz_ax(ax, method, distance_from_groundtruth):

        ax.set_title(method)
        ax.hist(distance_from_groundtruth[method].dropna(), bins = 100, alpha = 0.7)
        ax.set_yscale('log')
        ax.set_xlim([0,5])
        
    f, ax   = plt.subplots(2, 2, figsize=(15, 10))
    methods = distance_from_groundtruth.keys()

    print (methods[1])
    viz_ax(ax[0,0], methods[5], distance_from_groundtruth)    
    viz_ax(ax[0,1], methods[1], distance_from_groundtruth)    
    viz_ax(ax[1,0], methods[2], distance_from_groundtruth)    
    viz_ax(ax[1,1], methods[3], distance_from_groundtruth)    

    plt.savefig(outfolder   + '/user_homes/comparison/' + city + '_home_distance_from_groundtruth_' + str(LIMIT_num) + '.png')
    plt.close()
   


 
'''  ----------------------------------------------------------  '''
'''         viz the assumed homes dist from the groundtruth      '''
'''  ----------------------------------------------------------  '''

def get_final_comp_results(city, outfolder, LIMIT_num):

    groundtruth_homes         = get_groundtruth_homes(city, outfolder)
    methods_homes             = get_homes_from_methods(city, outfolder, LIMIT_num)
    distance_from_groundtruth = get_distance_from_groundtruth(methods_homes, groundtruth_homes, city, outfolder)

    

    df_res = pd.DataFrame()
    df_res['Averages'] = distance_from_groundtruth.mean(axis=0)
    df_res['Stdevs']   = distance_from_groundtruth.std(axis=0)

    print(df_res)

    df_res.to_csv(outfolder + '/user_homes/comparison/' + city + '_CENTROID_COMPARISON_RES_' + str(LIMIT_num) + '.csv', sep = '\t', float_format='%.3f')

    dummy_viz_goodness_of_dist(distance_from_groundtruth, city, outfolder, LIMIT_num)    
      
        

def plot_final_results(city, outfolder):

    files          = sorted([fff for fff in os.listdir(outfolder   + '/user_homes/comparison/') if 'csv' in fff])#+ city + '_CENTROID_COMPARISON_RES_' + str(LIMIT_num) + '.csv')
    methods_series = {}
    f, ax          = plt.subplots(1, 2, figsize=(12, 6))

    for fn in files:

        index = int(fn.split('_')[-1].split('.')[0])

        for line in open(outfolder +  '/user_homes/comparison/' + fn):
            if 'Averages' not in line:
        
                method, avg, std =  line.strip().split('\t')
                avg = float(avg)
                std = float(std)

                if method not in methods_series:
                    methods_series[method] = [(index, avg, std)]
                else:
                    methods_series[method].append((index, avg, std))
                

    
    for m, s in methods_series.items():

        s.sort(key=lambda tup: tup[0])

        ind, avg, std = zip(*s)
        ax[0].errorbar(ind, avg, yerr = std, fmt = 'o-', label = m)
        ax[1].plot(ind, avg, 'o-', label = m)


    ax[0].set_xlabel('Min. number of locations/user')
    ax[0].set_ylabel('Avg distance between estimated and real residence location [km]')
    ax[1].set_xlabel('Min. number of locations/user')
    ax[1].set_ylabel('Avg distance between estimated and real residence location [km]')
    ax[0].legend(loc = 'left', fontsize = 8)
    ax[1].legend(loc = 'left', fontsize = 8)
    plt.savefig(outfolder   + '/user_homes/' + city + '_home_distance_from_groundtruth_FULL.png')
    plt.show() 
    







