import os
import pandas as pd
import numpy as np
import mpu
from pyproj import Proj
from shapely.geometry import shape
from scipy.stats import entropy




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''
def get_venues_users(outfolder):

    venues_users = {}

    for ind, line in enumerate(open(outfolder + '/venues_info/london_venues_users.dat')):
        #if ind == 10: break
        fields = line.strip().split('\t')
        venue = fields[0]
        users = fields[1:]
        
        venues_users[venue] = users

    return venues_users




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''
def get_users_coordinates(outfolder, city):

    ### get users coordinates
    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    outroot   = '../ProcessedData/' + city + '/'
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'


    users_location = {}

    for line in open(infile):
        user, lng, lat = line.strip().split('\t')
        users_location[user] = (float(lng), float(lat))


    return users_location




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_venues_users_coordinates(relevant_venues, venues_users, users_location):

    venues_users_coordinates = {}

    relevant_venues = set(relevant_venues)

    for ind, (venue, users) in enumerate(venues_users.items()):
        
        if venue in relevant_venues:
        
            #if ind == 10: break
            for user in users:
                if user in users_location:
                    if venue not in venues_users_coordinates:
                        venues_users_coordinates[venue] = [users_location[user]]
                    else:
                        venues_users_coordinates[venue].append(users_location[user])
          

    return venues_users_coordinates




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_venues_coordinates(outfolder):

    venues_coordinates = {}

    for ind, line in enumerate(open(outfolder + '/venues_info/venues_ward_full.dat')):    
        if 'venue' not in line:       
            #if ind == 10: break
            venue, lng, lat, ward, lng0, lat0, lng1, lat1, length, area = line.strip().split('\t')      
            venues_coordinates[venue] = (float(lng), float(lat))        

    return venues_coordinates



'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_venues_geo_features(outfolder, venues_users_coordinates, venues_coordinates):


    venues_w_coordinate_users = []
    venues_geostretch_feats   = {}

    for ind, (venue, users) in enumerate(venues_users_coordinates.items()):
        
        venuelng, venuelat = venues_coordinates[venue]  
        venues_w_coordinate_users.append(len(users))
        distances          = [mpu.haversine_distance((userlat, userlng ), (venuelat, venuelng)) for (userlng, userlat) in users]
             
            
        social_stretch = 1.0 / len(distances) * sum(distances)
        social_std     = np.std(distances)
        
        
        polygon  = 0
        lat, lon = zip(*users)

        if len(lat) > 2:

            pa      = Proj("+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55")
            x, y    = pa(lon, lat)
            cop     = {"type": "Polygon", "coordinates": [zip(x, y)]}
            polygon = shape(cop).area / 1000000.0
        
        
        if venue not in venues_geostretch_feats:
            venues_geostretch_feats[venue] = {}
        venues_geostretch_feats[venue]['social_stretch'] = social_stretch
        venues_geostretch_feats[venue]['social_std']     = social_std
        venues_geostretch_feats[venue]['polygon']        = polygon

       
    df = pd.DataFrame.from_dict(venues_geostretch_feats, orient = 'index') #, columns = ['Llikecount'])
    df.to_csv(outfolder + 'venues_info/' + city + '_venues_geo_features.csv' , sep = '\t')    




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_users_venue_coordinates(outfolder, users_location):


    users_venue_coordinates = {}

    for ind, line in enumerate(open(outfolder+ 'user_info/london_FINAL_user_coordinates_raw_locals_filtered.dat')):

        #if ind == 10: break
            
        fields      = line.strip().split('\t')
        user        = fields[0]
        
        if user in users_location: 
            
            coordinates = [(float(fff.split(', ')[0]), float(fff.split(', ')[0])) for fff in fields[1:]]
            users_venue_coordinates[user] = coordinates

    return users_venue_coordinates




'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_users_geostretch_feats(users_venue_coordinates, users_location): 

    users_geostretch_feats = {}


    for ind, (user, venues) in enumerate(users_venue_coordinates.items()):

        #if ind == 10: break
        #print user, users_location[user], venues
        
        userlng, userlat = users_location[user]  
        distances        = [mpu.haversine_distance((venlat, venlng ), (userlat, userlng)) for (venlng, venlat) in venues]
             
            
        social_stretch   = 1.0 / len(distances) * sum(distances)
        social_std       = np.std(distances)
        
        
        polygon  = 0.0
        lat, lon = zip(*venues)

        if len(lat) > 2:

            pa      = Proj("+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55")
            x, y    = pa(lon, lat)
            cop     = {"type": "Polygon", "coordinates": [zip(x, y)]}
            polygon = shape(cop).area / 1000000.0
        
        
        if user not in users_geostretch_feats:
            users_geostretch_feats[user] = {}
        users_geostretch_feats[user]['social_stretch'] = social_stretch
        users_geostretch_feats[user]['social_std']     = social_std
        users_geostretch_feats[user]['polygon']        = polygon

    df = pd.DataFrame.from_dict(users_geostretch_feats, orient = 'index') #, columns = ['Llikecount'])
    df.to_csv(outfolder + 'venues_info/' + city + '_users_venues_geo_features.csv' , sep = '\t')    

    return users_geostretch_feats
    



'''  ============================================================== '''
'''  -------------------------------------------------------------- '''

def get_features_of_venues_from_users(venues_users, users_geostretch_feats):



    features_of_venues_from_users = {}

    for ind, (ven, users) in enumerate(venues_users.items()):
        
        #if ind == 100: break
            

        'social_stretch'
        'social_std'
        social_stretch = []
        social_std     = []
        polygon        = []
            
        for user in users:
            
            if user in users_geostretch_feats:
                #print ven, user, users_geostretch_feats[user]['polygon']
                social_stretch.append(users_geostretch_feats[user]['social_stretch'])
                social_std.append(users_geostretch_feats[user]['social_std'])
                polygon.append(users_geostretch_feats[user]['polygon'])            


        
        if len(social_stretch) == 0: social_stretch = [0]
        if len(social_std)     == 0: social_std     = [0]
        if len(polygon)        == 0: polygon        = [0]

            
        social_stretch = np.asarray(social_stretch)
        social_std     = np.asarray(social_std)
        polygon        = np.asarray(polygon)
            
        features_of_venues_from_users[ven] = {}    
            
        features_of_venues_from_users[ven]['social_stretch_avg']     = np.mean(social_stretch)
        features_of_venues_from_users[ven]['social_stretch_std']     = np.std(social_stretch)
        features_of_venues_from_users[ven]['social_stretch_entropy'] = entropy(social_stretch, base = len(social_stretch))  
        features_of_venues_from_users[ven]['social_std_avg']     = np.mean(social_std)
        features_of_venues_from_users[ven]['social_std_std']     = np.std(social_std)
        features_of_venues_from_users[ven]['social_std_entropy'] = entropy(social_std, base = len(social_std))
        features_of_venues_from_users[ven]['polygon_avg']     = np.mean(polygon)  
        features_of_venues_from_users[ven]['polygon_std']     = np.std(polygon)
        features_of_venues_from_users[ven]['polygon_entropy'] = entropy(polygon, base = len(polygon))  

    
    df = pd.DataFrame.from_dict(features_of_venues_from_users, orient = 'index') #, columns = ['Llikecount'])
    df.to_csv(outfolder + 'venues_info/' + city + '_users_AGGERG_venues_geo_features.csv' , sep = '\t')    





if __name__ == '__main__': 



    city            = 'london'
    outfolder       = '../ProcessedData/' + city + '/'
    relevant_venues = list(set([line.strip().split('\t')[0] for line in open(outfolder + '/venues_info/venues_ward_full.dat') if 'venue' not in line]))



    venues_users             = get_venues_users(outfolder)
    users_location           = get_users_coordinates(outfolder, city)
    venues_users_coordinates = get_venues_users_coordinates(relevant_venues, venues_users, users_location)
    venues_coordinates       = get_venues_coordinates(outfolder)
    get_venues_geo_features(outfolder, venues_users_coordinates, venues_coordinates)



    users_venue_coordinates = get_users_venue_coordinates(outfolder, users_location)
    users_geostretch_feats  = get_users_geostretch_feats(users_venue_coordinates, users_location)
    get_features_of_venues_from_users(venues_users, users_geostretch_feats)


