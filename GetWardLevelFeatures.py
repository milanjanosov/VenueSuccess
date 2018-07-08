import os
city      = 'london'
outfolder = '../ProcessedData/' + city + '/'
from collections import Counter
import math
import pandas as pd





def get_distr_venues(outfolder, resolution):

    filename        = outfolder + 'venues_info/venues_' + resolution + '_full.dat'
    ward_venues     = {}
    relevant_venues = set()
    
    for ind, line in enumerate(open(filename)):
        
        if 'ward' not in line:

            venue, lng, lat, ward, lng0, lat0, lng1, lat1, length, area = line.strip().split('\t')

            if ward not in ward_venues:
                ward_venues[ward] = [venue]
            else:
                ward_venues[ward].append(venue)

            relevant_venues.add(venue)

        #if ind == 10: break
        
      
    return ward_venues, relevant_venues





def venues_categories(outfolder, relevant_venues): 

    venues_cat    = {}
    venues_subcat = {}
    filename      =  outfolder + 'venues_info/venues_all_categories_times.dat'
    
    for ind, line in enumerate(open(filename)):
        #if ind == 10: break
        uer, venue, cat, subcat, idn = line.strip().split('\t')

        if venue in relevant_venues:
        
            if venue not in venues_cat:
                venues_cat[venue] = cat

            if venue not in venues_subcat:
                venues_subcat[venue] = subcat           
            
    
    return venues_cat, venues_subcat



  

def get_ward_categories(ward_venues, venues_cat, venues_subcat):

    all_cats    = []
    all_subcats = []

    
    ward_cat    = {}
    ward_subcat = {}

    for ind, (ward, venues) in enumerate(ward_venues.items()):

        #if ind == 10: break

        #print ward, len(venues)
        
        for venue in venues:

            if ward not in ward_cat:
                ward_cat[ward] = [venues_cat[venue]]
            else:
                ward_cat[ward].append(venues_cat[venue])

            if ward not in ward_subcat:
                ward_subcat[ward] = [venues_subcat[venue]]
            else:
                ward_subcat[ward].append(venues_subcat[venue])
                
            all_cats.append(venues_cat[venue])
            all_subcats.append(venues_subcat[venue])


    all_cat_types = list(set(all_cats))
    all_cat_freq  = Counter(all_cats)
            
    return ward_cat, ward_subcat, all_cat_types, all_cat_freq, all_cats





''' ADD THE  NUMBER OF USERS CENTROID WARD'''
def wards_user_centroids(outfolder):
    
    ward_users = {}
    
    #for line in open(outfolder + 'user_info/user_wards.dat'):
    
    for line in open(outfolder + 'user_info/user_wards.dat'):
        user, ward = line.strip().split('\t')
        
        if ward not in ward_users:
            ward_users[ward] = [user]
        else:
            ward_users[ward].append(user)

            
    ward_numc = {}
    
    for ward, users in ward_users.items():
        ward_numc[ward] = len(users)
        
    return ward_numc
       




def get_ward_features(outfolder, ward_cat, ward_subcat, all_cat_types, all_cat_freq, ward_venues, ward_numcentr, all_cats): 

    ward_cat_distr = {}

    # get P_ij for the entropy 
    # all_cat_prob = {}
    # sum_cats     = sum(list(all_cat_freq.values()))
    # for cat, num in all_cat_freq.items():
    #     all_cat_prob[cat] = all_cat_freq[cat]/float(sum_cats)


    entropies = []

    nnn = len(ward_cat)

    for ind, (ward, cats) in enumerate(ward_cat.items()):

        #if ind == 10: break
        print ind, '/', nnn

        num_of_venues = len(cats) 

        if ward not in ward_cat_distr:
            ward_cat_distr[ward] = {}

        # get the density of certain categories within wards
        E = 0

        for ac in all_cats:
            Pij = cats.count(ac) / float(num_of_venues)
            ward_cat_distr[ward][ac] = cats.count(ac) / float(num_of_venues)
            if Pij != 0:
                E -= Pij * math.log(Pij) / math.log(len(all_cats))

        ward_cat_distr[ward]['Entropy']    = E
        ward_cat_distr[ward]['Venues_num'] = num_of_venues


    venues_ward_cat_stats = {}
        

    nnn = len(ward_venues)
        
    for ind, (ward, venues) in enumerate(ward_venues.items()):
        
        #if ind == 10: break
        print ind, '/', nnn
        
        for venue in venues:
            venues_ward_cat_stats[venue] = ward_cat_distr[ward]
            venues_ward_cat_stats[venue]['ward'] = ward
            venues_ward_cat_stats[venue]['user_centroids'] = ward_numcentr[ward]

        

    df = pd.DataFrame.from_dict(venues_ward_cat_stats, orient = 'index')
    df.to_csv(outfolder + 'venues_info/' + city + '_WARD_category_stats.csv' , sep = '\t')        
    
    




ward_venues, relevant_venues = get_distr_venues(outfolder, 'ward')
venues_cat, venues_subcat    = venues_categories(outfolder, relevant_venues)
       
ward_cat, ward_subcat, all_cat_types, all_cat_freq, all_cats = get_ward_categories(ward_venues, venues_cat, venues_subcat)

ward_numcentr = wards_user_centroids(outfolder)        
get_ward_features(outfolder, ward_cat, ward_subcat, all_cat_types, all_cat_freq, ward_venues, ward_numcentr, all_cats)









