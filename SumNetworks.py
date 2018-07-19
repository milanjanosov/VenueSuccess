import os
import pandas as pd
import scipy.stats as stat
import numpy as np
from multiprocessing import Process





city      = 'london'
outroot   = '../ProcessedData/' + city 



threshold = 5000

for threshold in [5000, 3000, 2000, 1500, 1000]:#, 500, 100]:

#if 2 == 2:

    users_friend_geo   = outroot + '/networks/' + city + '_friend__networkmeasures.csv'
    venues_sim         = outroot + '/networks/' + city + '_venues_similarity_NC_' + str(threshold) + '_networkmeasures.csv'
    users_sim          = outroot + '/networks/' + city + '_users_sim_geo_NC_'     + str(threshold) + '_networkmeasures.csv'
    relevant_venues    = set([line.strip().split('\t')[0] for line in open(outroot + '/venues_info/venues_ward_full.dat') if 'venue' not in line])


    df_friend_geo  =  pd.read_csv(users_friend_geo,  sep = ',', index_col=0).fillna(0.0) 
    df_user_sim    =  pd.read_csv(users_sim,         sep = ',', index_col=0).fillna(0.0) 
    df_venue_sim   =  pd.read_csv(venues_sim,        sep = ',', index_col=0).fillna(0.0) 


    venues_features = {}
    venues_user     = outroot + '/venues_info/' + city + '_venues_users.dat'
    venues_users    = {}




    for line in open(venues_user):
        field = line.strip().split('\t')    
        venue = field[0]
        users = field[1:]

        if venue in relevant_venues:
            venues_users[venue] = users





    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'
      
    users_c   = [int(line.strip().split('\t')[0]) for line in open(infile)]




    #print df_friend_geo

    nnn = len(venues_users)

    for ind, (v, users) in enumerate(venues_users.items()):

        #if ind == 50: break


        venues_features[v] = {}


        for (sign, df) in [('u', df_friend_geo), ('us', df_user_sim)]:


            keys_geo  = df.keys()
            users_nw  = list(df.index.values)
            users     = [int(u) for u in users if int(u) in users_nw]
           
            
            print ind,'/', nnn

            for key in keys_geo:

                user_level_values = df.loc[users][key].tolist()
                nmn = len(np.asarray(user_level_values))
                user_level_values = [ijk/nmn for ijk in user_level_values]

                if 'geo_size_of_ego' in key:
                    user_level_values = [u / 1000000.0 for u in user_level_values]


                if len(user_level_values) > 0:
                    venues_features[v][sign + '_' + key + '_avg']     = np.mean(user_level_values)
                    venues_features[v][sign + '_' + key + '_std']     = np.std(user_level_values)
                    venues_features[v][sign + '_' + key + '_entropy'] = stat.entropy(np.asarray(user_level_values), base = len(user_level_values))
                else:
                    venues_features[v][sign + '_' + key + '_avg']     = 0
                    venues_features[v][sign + '_' + key + '_std']     = 0
                    venues_features[v][sign + '_' + key + '_entropy'] = 0
            






    filename = outroot + '/networks/' + city + '_COMBINED_networkmeasures_' + str(threshold) + '.csv'
    df = pd.DataFrame.from_dict(venues_features, orient = 'index')


    df_venue_sim = df_venue_sim.rename(index=str, columns = {vv : 'v_' + vv for vv in df_venue_sim.keys()})


    finaldf = df.join(df_venue_sim)

    finaldf.to_csv(filename, na_rep='nan')















