import sys
sys.path.append("..")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import ParseInput
import ParseJsons 

import GetHomeLocations as Home
import MachineLearningHomeFeatures as MLFeat
import WhereIsHomeClassification as Class
import CompareHomingMethods as Compare
import time
import FilterHomeLocatoins as FilterH
import OptimizeDBScan as Optimize
import subprocess
#import SumNetworks as SNW





''' =========================================================== '''
''' ===================   ParseInput.py   ===================== '''
''' =========================================================== '''

inputs = ParseInput.get_inputs()


city  = sys.argv[1]
bbox  = inputs[city]
server = False

#if len(sys.argv == 4):
#    matplotlib.use('Agg')
#    server = True
#import matplotlib.pyplot as plt


inroot  = '../Data/fsqdb/'    + city + '/'
outroot = '../ProcessedData/' + city + '/'


ParseJsons.create_folder(outroot + 'user_info') 
ParseJsons.create_folder(outroot + 'basic_stats') 
ParseJsons.create_folder(outroot + 'venues_info')
ParseJsons.create_folder(outroot + 'user_homes')
ParseJsons.create_folder(outroot + 'networks')
ParseJsons.create_folder(outroot + 'networks/gephi')
ParseJsons.create_folder(outroot + 'figures/user_homes')
ParseJsons.create_folder(outroot + 'figures/network_data')
ParseJsons.create_folder(outroot + 'user_homes/comparison')
ParseJsons.create_folder(outroot + 'user_homes/MLfeatures')
ParseJsons.create_folder(outroot + 'user_homes/MLresults')
ParseJsons.create_folder(outroot + 'user_homes/MLhomes')
ParseJsons.create_folder(outroot + 'user_homes/centroids')
ParseJsons.create_folder(outroot + 'user_homes/optimize_centroids/')
ParseJsons.create_folder(outroot + 'user_homes/optimize_centroids/series')
ParseJsons.create_folder(outroot + 'figures')
ParseJsons.create_folder(outroot + 'figures/MLresults')





if sys.argv[2] == 'preproc':

    
    ''' =========================================================== '''
    ''' ===================   ParseJsons.py   ===================== '''
    ''' =========================================================== '''

    t1 = time.time()


    users_homes   = {}

    unknown_users, local_users, nonlocal_users = ParseJsons.get_local_users(city, inroot, outroot)

#    users_likes   = ParseJsons.get_users_like_location(         unknown_users, local_users, city, bbox, inroot, outroot, users_homes) 
   # users_photos  = ParseJsons.get_photos_locations_and_users(  unknown_users, local_users, city, bbox, inroot, outroot, users_homes)
    users_tips    = ParseJsons.get_tips_locations_and_users(    unknown_users, local_users, city, bbox, inroot, outroot, users_homes)

    ParseJsons.write_home_locations( users_homes, city, outroot,  len(users_photos.keys()))

    ParseJsons.get_users_coordinates(users_homes, local_users, {}, users_tips, {}, city, outroot, bbox)
    ParseJsons.get_users_distance_distr_from_home(city, outroot) 
    ParseJsons.get_users_venues(unknown_users, local_users, users_photos, users_likes, users_tips, city, outroot)

    ParseJsons. get_users_friends(local_users, city, inroot, outroot)

    ParseJsons.venues_distance_mtx(bbox, city, outroot)
    ParseJsons.get_venues_users(city, outroot)       


    t2 = time.time()

    print ('Preprocess time: ', t2 - t1)   




 






elif sys.argv[2] == 'home_sample':

    ''' this sample stuff creates vizs '''



    ''' REWRITE ERRE: _user_venues_full_locals_filtered   '''


    for LIMIT in [0, 10]:

        t1 = time.time()

        user_sample = Home.get_users_centroids(           city, outroot, sample = True, LIMIT_num = LIMIT,              plot = False)
   #     Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 1.0, plot = False)
        Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 2.0, plot = False)
   #     Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 3.0, plot = False)
   #     Home.get_db_centroids(               user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, eps = 0.01,  mins = 3)
        Home.get_db_centroids(               user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, eps = 0.02,  mins = 3)
   #     Home.get_db_centroids(               user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, eps = 0.1,   mins = 3)


    t2 = time.time()

    print ('Sample home clustering with plots: ', t2 - t1)   




elif sys.argv[2] == 'home_full':


    t1 = time.time()

    for LIMIT in range(20):

        print ('LIMIT = ' + str(LIMIT))

        ''' these are the centroid based things '''
        users = Home.get_users_centroids(           city, outroot, sample = False, LIMIT_num = LIMIT,              plot = False)

        Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 0.5, plot = False)
        Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 1.0, plot = False)
        Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 2.0, plot = False)
        Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 5.0, plot = False)
        Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.01, mins = 3)
        Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.02, mins = 3)
        Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.02, mins = 5)
        Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.1,  mins = 3)
        Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.2,  mins = 3)

        ''' this is the messy ML part '''
        MLFeat.gennerate_classification_features(city, outroot, LIMIT, N = 4, R = 1.0)
        Class.classify_data(city, outroot, LIMIT)
        Class.conclude_class(city, outroot, LIMIT)

        
     # this has to be run only once after that loop above
    FilterH.copy_filtered(city, outroot, bbox)

    for LIMIT in range(20):   Compare.get_final_comp_results(city, outroot, LIMIT_num = LIMIT)
    ''' this compares the different methods '''
    
    Compare.plot_final_results(city, outroot)



    t2 = time.time()
    print ('Full home clustering with plots: ', t2 - t1)   



elif sys.argv[2] == 'opt_dbscan':

    Optimize.optimize_db_scan(city, outroot)
    Optimize.visualize_opt_res(city, outroot)


    

elif sys.argv[2] == 'networks':

    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    infile    = outroot + '/user_homes/centroids/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '.dat'
   


    call_python_version("2.7", "BuildNetworks", "do_all_the_networks", [city, outroot, infile, bbox])


    


elif sys.argv[2] == 'success' : 

    a = 0
    ###    issue: from venues.json, photos, tips, and likes, we wont get the same venues
    ###
    ###    ParseJsons.get_venues_information(city, bbox, inroot, outroot)   






