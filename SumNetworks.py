import os
import pandas as pd
import scipy.stats as stat
import numpy as np


city        = 'bristol'
outroot     = '../ProcessedData/' + city 

users_friend_geo   = outroot + '/networks/' + city + '_users_geo_networkmeasures.csv'
users_friend_topo  = outroot + '/networks/' + city + '_users_topo_networkmeasures.csv'
users_sim          = outroot + '/networks/' + city + '_users_topo_networkmeasures.csv'
venues_sim         = outroot + '/networks/' + city + '_venues_sim_geo_networkmeasures.csv'


df_friend_geo  =  pd.read_csv(users_friend_geo,  sep = ',', index_col=0) 
df_friend_topo =  pd.read_csv(users_friend_topo, sep = ',', index_col=0) 
df_user_sim    =  pd.read_csv(users_sim,         sep = ',', index_col=0) 
df_venue_sim   =  pd.read_csv(venues_sim,        sep = ',', index_col=0) 



venues_features = {}
venues_user     = outroot + '/venues_info/' + city + '_venues_users.dat'
venues_users    = {}




for line in open(venues_user):
    field = line.strip().split('\t')    
    venue = field[0]
    users = field[1:]

    venues_users[venue] = users


eps       = 0.02
mins      = 3
LIMIT_num = 5
infile    = outroot + '/user_homes/centroids/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '.dat'
  
users_c   = [int(line.strip().split('\t')[0]) for line in open(infile)]
users_nw  = list(df_friend_geo.index.values)



for ind, (v, users) in enumerate(venues_users.items()):


    if ind == 5: break

    if 2 == 2:


        keys_geo = df_friend_geo.keys()
        for key in keys_geo:
        

            users = [int(u) for u in users if int(u) in users_nw]
       
            #print(users, df_friend_geo.loc[users].head())#[key].tolist() )
            user_level_values = df_friend_geo.loc[users][key].tolist()
        

            if len(user_level_values) > 0:
                venues_features[key + '_avg']     = np.mean(user_level_values)
                venues_features[key + '_std']     = np.std(user_level_values)
                venues_features[key + '_entropy'] = stat.entropy(np.asarray(user_level_values))
            else:
                venues_features[key + '_avg']     = 0
                venues_features[key + '_std']     = 0
                venues_features[key + '_entropy'] = 0


    



        '''keys_topo = users_friend_topo.keys()
        for key in keys_topo:

            user_level_values                 = users_friend_topo.loc[users][key].tolist()
            venues_features[key + '_avg']     = np.mean(user_level_values)
            venues_features[key + '_std']     = np.avg(user_level_values)
            venues_features[key + '_entropy'] = stat.entropy(np.asarray(values))


        keys_sim = df_user_sim.keys()
        for key in df_friend_sim:

            user_level_values                 = df_user_sim.loc[users][key].tolist()
            venues_features[key + '_avg']     = np.mean(user_level_values)
            venues_features[key + '_std']     = np.avg(user_level_values)
            venues_features[key + '_entropy'] = stat.entropy(np.asarray(values))
        '''

 



print(venues_features)


'''1. VENUES_FEATURES --> DF
2. MERGE df_venue_sim and THIS DF
'''
















