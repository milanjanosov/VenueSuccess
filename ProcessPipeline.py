import sys
sys.path.append("..")
import FourSquareFeatureMiner.ParseInput as ParseInput
import FourSquareFeatureMiner.ParseJsons as ParseJsons
import GetHomeLocations as Home
import MachineLearningHomeFeatures as MLFeat
import WhereIsHomeClassification as Class
import CompareHomingMethods as Compare
import CallOldPython as Call 
import SumNetworks as SNW


''' TODO '''

'''
- classification confidence range! rank
- network of regions?

- MLFeat.gennerate_classification_features --> ADD LIMIT_num as the min num of posts
- dist distribution of users with home: home dist vs other dist, within box


'''



''' =========================================================== '''
''' ===================   ParseInput.py   ===================== '''
''' =========================================================== '''

inputs = ParseInput.get_inputs()


city  = sys.argv[1]
bbox  = inputs[city]


inroot  = '../Data/fsqdb/'    + city + '/'
outroot = '../ProcessedData/' + city + '/'


ParseJsons.create_folder(outroot + 'user_info') 
ParseJsons.create_folder(outroot + 'venues_info')
ParseJsons.create_folder(outroot + 'user_homes')
ParseJsons.create_folder(outroot + 'networks')
ParseJsons.create_folder(outroot + 'user_homes/figures')
ParseJsons.create_folder(outroot + 'user_homes/comparison')
ParseJsons.create_folder(outroot + 'user_homes/centroids')


# parse and preprocess the data

if sys.argv[2] == 'preproc':

    
    ''' =========================================================== '''
    ''' ===================   ParseJsons.py   ===================== '''
    ''' =========================================================== '''


    users_homes   = {}
  #  users_likes   = ParseJsons.get_users_like_location(city, bbox, inroot, outroot, users_homes)  
 #   num_users     = len(users_likes.keys())
   # users_friends = ParseJsons.get_users_friends(city, inroot, outroot)
 #   users_photos  = ParseJsons.get_photos_locations_and_users(city, bbox, inroot, outroot, users_homes)


  #  ParseJsons.write_home_locations(users_homes, city, outroot, num_users)#10)
  #  ParseJsons.get_users_coordinates(users_likes, users_friends, users_photos, city, outroot)
  #  ParseJsons.get_users_distance_distr_from_home(city, outroot)

    ParseJsons.get_users_venues(users_photos, users_likes, city, outroot)

    ParseJsons.get_venues_information(city, bbox, inroot, outroot)
 #   ParseJsons.venues_distance_mtx(city, outroot)

 #   ParseJsons.get_venues_users(city, outroot)



elif sys.argv[2] == 'home_sample':

    ''' this sample stuff creates vizs '''
    LIMIT = 10

    user_sample = Home.get_users_centroids(           city, outroot, sample = True, LIMIT_num = LIMIT,              plot = False)
    Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 0.5, plot = False)
    Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 1.0, plot = False)
    Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 2.0, plot = False)
    Home.get_users_centroids_with_cutoff(user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, limit = 5.0, plot = False)
    Home.get_db_centroids(               user_sample, city, outroot, sample = True, LIMIT_num = LIMIT, eps = 0.02,  mins = 3)


elif sys.argv[2] == 'home_full':

    LIMIT = 10

    for LIMIT in range(20):


        ''' these are the centroid based things '''
 #       users = Home.get_users_centroids(           city, outroot, sample = False, LIMIT_num = LIMIT,              plot = False)
 #       Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 0.5, plot = False)
 #       Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 1.0, plot = False)
 #       Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 2.0, plot = False)
 #       Home.get_users_centroids_with_cutoff(users, city, outroot, sample = False, LIMIT_num = LIMIT, limit = 5.0, plot = False)
 #       Home.get_db_centroids(users, city, outroot, sample = False, LIMIT_num = LIMIT,eps = 0.02, mins = 3)
        

        ''' this is the messy ML part '''
        #MLFeat.gennerate_classification_features(city, outroot, N = 4, R = 1.0)
        #Class.classify_data(city, outroot)
        #Class.conclude_class(city, outroot)

        ''' this compares the different methods '''
 #       Compare.get_final_comp_results(city, outroot, LIMIT_num = LIMIT)
    
    Compare.plot_final_results(city, outroot)


elif sys.argv[2] == 'networks':

    eps       = 0.02
    mins      = 3
    LIMIT_num = 5
    infile    = outroot + '/user_homes/centroids/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '.dat'
   

    Call.call_python_version('2.7', 'BuildNetworks', 'do_all_the_networks', [city, outroot, infile])




''' 

    MERGELES: 
    - venue -> zipcode map
    - full_locationt atmappelni
    - networkos reszt atparameterezni ugyh venue, zipcode, ...

'''









