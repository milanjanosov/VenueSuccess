import numpy as np
from collections import Counter
import pandas as pd




'''  ---------------------------------------------------------  '''
'''          parse all the venues a user interacted with        '''
'''  ---------------------------------------------------------  '''

def get_user_venues(city, outfolder):

    
    user_venues       = {}
    venues_categories = {}
    all_users         = []
    
    for line in open(outfolder + '/user_info/' + city + '_user_venues_full.dat'):
        if 'userid' not in line:
            fields = line.strip().split('\t')
            user   = fields[0]
            
            
            for field in fields[1:]:
                
                dields = field.split(',')
                venue  = dields[0]
                lng    = float(dields[1])
                lat    = float(dields[2])
                categ  = dields[3]
                
                if venue not in venues_categories:
                    venues_categories[venue] = categ
                
            
                if user not in user_venues:
                    user_venues[user] = [(venue, lng, lat, categ)]
                else:
                    user_venues[user].append((venue, lng, lat, categ))
                    
            
    return user_venues, venues_categories




'''  ----------------------------------------------------------  '''
'''  get the users' locations sorted by their distance from home '''
'''  ----------------------------------------------------------  '''

def get_venues_distances_sorted(city, outfolder, venues_categories):


    venues_distances = {}

    for index, line in enumerate(open(outfolder + '/venues_info/' + city + '_venues_distance_matrix.dat')):
        v1, v2, dist = line.strip().split('\t')
        dist = float(dist)
        
        
        if v1 not in venues_distances:
            venues_distances[v1] = {}
        
        venues_distances[v1][v2] = dist
            
        
        if v2 not in venues_distances:
            venues_distances[v2] = {}
        
        venues_distances[v2][v1] = dist    


         
    venues_distances_sorted = {}        
            
    for v, vs in venues_distances.items():
        venues_distances_sorted[v] = sorted([(ke, va, venues_categories[ke]) for ke, va in vs.items()], key=lambda tup: tup[1])


    return venues_distances_sorted     




'''  ---------------------------------------------------------  '''
'''          derive the venues features with two params         '''
'''          N: avg distance of the closest N places            '''
'''          R: avg number of places within R, close nghbhood   '''
'''  ---------------------------------------------------------  '''

def get_venues_features(city, outfolder, venues_distances_sorted, N = 4, R = 1.1):

    venues_features = {}

    for v0, venues in venues_distances_sorted.items():
        
        first_N_avg_dist    = np.mean([ v1[1] for index, v1 in enumerate(venues) if index < N ])
        first_N_categories  = Counter([ v1[2] for index, v1 in enumerate(venues) if index < N ])    

        within_R_categories = Counter([ v1[2] for index, v1 in enumerate(venues) if v1[1]  < R ])   
      
        venues_features[v0] = {
            'first_N_avg_dist'    :  first_N_avg_dist,
            'within_R_avg_dist'   :  within_R_avg_dist
        }
       

        for k, v in first_N_categories.items():
            venues_features[v0][k] = v
        
        for k, v in within_R_categories.items():
            venues_features[v0][k] = v
            


    return venues_features        




'''  ----------------------------------------------------------  '''
'''                         parse the homes                      '''
'''  ----------------------------------------------------------  '''

def get_users_home(city, outfolder):


    users_home = {} 

    for line in open(outfolder + '/user_info/' + city + '_groundtruth_home_locations_unique.dat'):
        user, lng, lat, venue = line.strip().split('\t')
        
        if user not in users_home:
            users_home[user] = venue
    

    return users_home
        



'''  ----------------------------------------------------------  '''
'''  get the final features                                      '''
'''  ----------------------------------------------------------  '''

def get_users_features(city, outfolder, venues_features, user_venues, users_home):

    users_features_with_home = []
    users_features_no_home   = []



    for user, venues in user_venues.items():
        
        
        has_home = 0
        home     = 0

        
        if user in users_home:
            has_home = True
            home     = users_home[user]
        else:
            has_home = False
        
        for venue in venues:
                     
            if venue[0] in venues_features:
                

                stuff = {}    
                stuff['user']  = user
                stuff['venue'] = venue[0]

                if has_home and venue[0] == users_home[user]:
                    stuff['home'] = 1
                else:
                    stuff['home'] = 0

                
                for k, v in venues_features[venue[0]].items():
                    stuff[k] = v
                    
                if has_home:
                    users_features_with_home.append(stuff)
                else:
                    users_features_no_home.append(stuff)                    

            
    df_home   = pd.DataFrame(users_features_with_home)
    df_nohome = pd.DataFrame(users_features_no_home)

    df_home.to_csv(outfolder   + '/user_homes/' + city + '_ML_feature_has_home_TRAIN.csv', sep = '\t')
    df_nohome.to_csv(outfolder + '/user_homes/' + city + '_ML_feature_has_home_TEST.csv' , sep = '\t')




def gennerate_classification_features(city, outfolder, N = 4, R = 1.0):

    

    user_venues, venues_categories = get_user_venues(city, outfolder)
    venues_distances_sorted        = get_venues_distances_sorted(city, outfolder, venues_categories)
    venues_features                = get_venues_features(city, outfolder, venues_distances_sorted, N, R )
    users_home                     = get_users_home(city, outfolder)

    get_users_features(city, outfolder, venues_features, user_venues, users_home)



