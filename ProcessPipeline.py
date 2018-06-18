import sys
sys.path.append("..")
import FourSquareFeatureMiner.ParseInput as ParseInput
import FourSquareFeatureMiner.ParseJsons as ParseJsons
import GetHomeLocations as Home
import MachineLearningHomeFeatures as MLFeat
import WhereIsHomeClassification as Class
import CompareHomingMethods as Compare
import CallOldPython as Call 
import time
#import SumNetworks as SNW


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

    unknown_users, local_users, nonlocal_users = ParseJsons.get_local_users(city, inroot, outroot)

    ''' three types of users:
            - live in bristol
            - dont live in bristol
            - we dont know
    '''

    t1 = time.time()
 
    users_likes   = ParseJsons.get_users_like_location(         unknown_users, local_users, city, bbox, inroot, outroot, users_homes) 
    users_photos  = ParseJsons.get_photos_locations_and_users(  unknown_users, local_users, city, bbox, inroot, outroot, users_homes)
    users_tips    = ParseJsons.get_tips_locations_and_users(    unknown_users, local_users, city, bbox, inroot, outroot, users_homes)


    print(len(users_likes), len(users_photos), len(users_tips))

    uuu = list(users_likes.keys()) + list(users_photos.keys()) + list(users_tips.keys())

    print(len(set(uuu).intersection(set(nonlocal_users))))

    ParseJsons.get_users_coordinates(nonlocal_users, users_likes, users_tips,  users_photos, city, outroot)
    ParseJsons.write_home_locations( users_homes, city, outroot,  len(users_likes.keys()))#10)
        
 
    
    t2 = time.time()

    print (t2 - t1)   


    ''' out_test:
        - local users shouldn't have venues outside of the bbox
        - nonlocal users shouldn't be present in the data
        - unknown users should have locations both within and outside of the bbox
    '''


    '''

    ParseJsons.get_users_distance_distr_from_home(city, outroot)           -->  csak amiknek van groundtruth home location, make sure fig is saved
    ParseJsons.get_users_venues(users_photos, users_likes, city, outroot)  -->  for local+unknown users , not sure why would i need this
    ParseJsons.get_venues_information(city, bbox, inroot, outroot)         -->  for all the venues we have here -- COMPARE venues.json AND FULL-VENUES FROM USERS
    ParseJsons.venues_distance_mtx(city, outroot)                          -->  ONLY VENUES WITHIN BBOX
    ParseJsons.get_venues_users(city, outroot)                             -->  ONLY venues within bbox and local_unknown users
    '''

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

    for LIMIT in range(1):


        ''' these are the centroid based things '''
        users = Home.get_users_centroids(           city, outroot, sample = False, LIMIT_num = LIMIT,              plot = False)
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
    1. FINISH SUMNETWORKS WHEN I HAVE THE NW MEASURES


    MERGELES: 
    - venue -> zipcode map
    - full_locationt atmappelni
    - networkos reszt atparameterezni ugyh venue, zipcode, ...

    '''









