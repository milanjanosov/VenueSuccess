import numpy as np
import random
import pandas as pd
import os
from xgboost import plot_importance
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
import sys
warnings.filterwarnings('ignore')






def get_features():



    features = pd.read_csv('senior_venues_features_labels.csv', sep = '\t', index_col = 0)

    user_nw   = [u for u in features.keys() if 'WHO_u' in u]
    user_succ = [u for u in features.keys() if 'tipCount' in u or 'checkinsCount' in u or 'usersCount' in u or 'Llikecount' in u or 'lLondon' in u]
    features  = features.drop(columns = user_nw + user_succ)
    features  = features.drop(columns = [c for c in features.keys() if '_inv_' in c])
    features  = features.drop(columns = [c for c in features.keys() if '_grav_' in c])
    features  = features.dropna()

    features  = features.drop(columns = ['WHO_v_social_stretch', 'WHO_v_social_std'])
    features2 = features

    firstimes = pd.read_csv('venues_firstime.csv', sep = '\t', header = None, index_col = 0)
    firstimes.index.name = 'venue'
    firstimes = firstimes.rename(columns = {1: 'WHEN_firsttime', 2 :'WHEN_hour_most', 3 : 'WHEN_day_most'})
    firstimes = firstimes.drop(columns = [4])

    genders = pd.read_csv('venues_genders.csv', sep = '\t',  index_col = 0)
    genders = genders.rename(columns = {'female': 'WHO_female_fraction'})
    genders = genders.drop(columns = ['LABEL_category'])

    genders.head()

    features2 = features2.join(firstimes)
    features2 = features2.join(genders)

    print len(features2.keys()), len(features2)
    features2.head()
    features2.head()
    firstimes.head()





    outfolder  = 'Figures/datafiles/'
    files      = [outfolder + f for f in os.listdir(outfolder) if 'clusters_data_all_' in f]
    indicies   = [(i,j) for i in range(2) for j in range(3)]
    clusters   = [fn.split('_')[3].replace('.dat', '') for fn in files] 
    N          = 160

    clusters_venues = {}
    for cluster in clusters:
        clusters_venues[cluster] = []
        for ind, line in enumerate(open(outfolder + '/sorted_dist/sorted_dist_venues_in_' + cluster + '.dat')):
            if ind == N: break
            clusters_venues[cluster].append(line.strip().split('\t')[0])


    top100venues = []
    for c, vens in clusters_venues.items():
        
        top100venues += vens
        

    print clusters_venues.keys()
    features3 = features2[features2.index.isin(top100venues)]    
    print len(features), len(features2), len(features3.keys()), len(features3)

    features3 = features3.dropna()
    features2.to_csv('FINAL_FEATURES_UNCORR_BALANCED_ALL.csv', '\t')
    features3.to_csv('FINAL_FEATURES_UNCORR_BALANCED_160.csv', '\t')









    X = features3.drop(columns = ['LABEL_category'])
    y = np.asarray(features3.LABEL_category)






    features_most_pred = ['WHEN_firsttime', 
                          'WHEN_hour_most',
                          'WHAT_venue_cat', 
                          'WHAT_pricerange',
                        'WHEN_firsttime',
                        'WHO_m_entropy',
                        'WHERE_emb_own_cat',
                        'WHO_fraction_of_regulars',
                        'WHERE_emb_na',
                        'WHO_m_std',
                        'WHO_m_avg',
                        'WHERE_emb_own_cat',
                        'WHERE_emb_na',
                        'WHERE_distance_from_center',
                        'WHERE_emb_building',
                        'WHERE_emb_travel',
                        'WHERE_emb_event',
                        'WHO_m_entropy',
                        'WHO_female_fraction',
                        'WHO_fraction_of_regulars',
                        'WHO_m_std',
                        'WHO_m_3_fraction',
                        'WHO_m_avg']




    redundant_features = ['WHERE_emb_na', 'WHO_m_3_fraction', 'WHO_m_avg', 'WHO_m_std','WHERE_emb_event', 'WHERE_emb_travel']
    features_most_pred_uncorr = [f for f in features_most_pred if f not in redundant_features]
    X_pred_uncorr = X.drop(columns = [c for c in X.keys() if c not in features_most_pred_uncorr])



    features_most_pred = list(set(features_most_pred))
    network_users      = pd.DataFrame.from_csv('FINAL_DATA/london_COMBINED_networkmeasures_2000.csv')
    network_users      = network_users.drop(columns = [c for c in network_users.keys() if c != 'u_pagerank_avg'])
    category_distances = pd.DataFrame.from_csv('../../ProcessedData/london/venues_info/CATEGORY_embeddedness.dat', sep = '\t')
    category_distances = category_distances[category_distances.index.isin(X_pred_uncorr.index)]




    X_pred_uncorr_renamed = pd.DataFrame()


    X_pred_uncorr_renamed['WHERE - Bohemian score']         = X['WHERE_ArtsEmploy']
    X_pred_uncorr_renamed['WHERE - IMD score']              = X['WHERE_IMDScore']
    X_pred_uncorr_renamed['WHERE - Population density']     = X['WHERE_PopDen']
    X_pred_uncorr_renamed['WHERE - Residential neighbourhood']   = category_distances['building']
    X_pred_uncorr_renamed['WHAT - Unique in neighbourhood']    = category_distances['own_cat']
    X_pred_uncorr_renamed['WHAT - Venue category']             = X_pred_uncorr['WHEN_firsttime']
    X_pred_uncorr_renamed['WHAT - Price category']             = X_pred_uncorr['WHEN_hour_most']
    X_pred_uncorr_renamed['WHO - Regulars score']                  = X_pred_uncorr['WHO_fraction_of_regulars']
    X_pred_uncorr_renamed['WHO - Diversity in purchasing power']   = X_pred_uncorr['WHO_m_entropy']

    aa = pd.read_csv('venues_genders.csv', sep = '\t', index_col = 0)
    aa = aa.rename(columns = {'female' : 'WHO - Gender balance score'})

    X_pred_uncorr_renamed = X_pred_uncorr_renamed.merge(network_users, right_index = True, left_index = True)
    X_pred_uncorr_renamed = X_pred_uncorr_renamed.merge(aa, right_index = True, left_index = True)
    X_pred_uncorr_renamed = X_pred_uncorr_renamed.rename(columns = {'LABEL_category' : 'Cluster'})
    X_pred_uncorr_renamed = X_pred_uncorr_renamed.rename(columns = {'u_pagerank_avg' : 'WHO - Average Pagerank of customers from Contact Network' })
    X_pred_uncorr_renamed_Labeled = X_pred_uncorr_renamed





    XNEW = X_pred_uncorr_renamed_Labeled.drop(columns = ['Cluster'])
    yNEW = np.asarray(X_pred_uncorr_renamed_Labeled.Cluster)


    return XNEW, yNEW







def xgb_model_params_types_given_features(X, y, max_depth_ ,learning_rate_, subsample_, tipus = ''):
              
    X_ = X.drop(columns = [c for c in X.keys() if tipus not in c])  
     
    #print X_.keys()
        
    #print len(X_.keys())
    train_data, test_data, train_label, test_label =  train_test_split(X_, y, test_size=.33, random_state=42)    
          
    model2       = xgb.XGBClassifier(n_estimators=100, max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
    train_model2 = model2.fit(train_data, train_label)
    pred2        = train_model2.predict(test_data)
            
    accuracies = list(cross_val_score(train_model2, train_data, train_label, cv=5))       

        
    return (np.mean(accuracies), np.std(accuracies))
     
    








XNEW, yNEW = get_features()


feattype = sys.argv[1]



folderout = 'FINAL_OPT_RES'
if not os.path.exists(folderout): os.makedirs(folderout)




fout = open(folderout + '/' + feattype + '.dat', 'w')
nout = open(folderout + '/' + feattype + '_best.dat', 'w')


best = [0,0,0,0]

for depth in [4, 5,6][0:2]:

    for rate in [0.01, 0.05, 0.1, 0.15, 0.2][0:2]:

    
        for sample in [0.7, 0.85, 0.9, 0.95][0:2]: 



                                       
            All,    Aerrall   = xgb_model_params_types_given_features(XNEW, yNEW, depth, rate, sample, tipus = feattype )
            #Awhat,  Aerrwhat  = xgb_model_params_types_given_features(XNEW, yNEW, depth, rate, sample, tipus = 'WHAT' )
            #Awho,   Aerrwho   = xgb_model_params_types_given_features(XNEW, yNEW, depth, rate, sample, tipus = 'WHO'  )
            #Awhere, Aerrwhere = xgb_model_params_types_given_features(XNEW, yNEW, depth, rate, sample, tipus = 'WHERE')

            fout.write( feattype + '\t' +  str(depth) + '\t' + str(rate) + '\t' + str(sample) + '\t' +str(All) + '\n')

            if All > best[0]:
                best = [All, depth, rate, sample]
        

fout.close()

nout.write(  '\t'.join( [str(fff) for fff in best ] ) )

nout.close()






