# https://developer.foursquare.com/docs/api/photos/details 
import os, sys
import json
import numpy as np
import csv
import mpu
import gzip
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter



'''  ---------------------------------------------------------  '''
'''  make sure that the coordinates are within the bounding box '''
'''  ---------------------------------------------------------  '''

def check_box(boundingbox, city, lat, lng):

    isin = True

    if (lat < min(boundingbox[0:2]) or lat > max(boundingbox[0:2])) or (lng < min(boundingbox[2:]) or lng > max(boundingbox[2:])):
        isin = False
    
    return isin




'''  ---------------------------------------------------------  '''
'''  harness the locations of places which the users liked      '''
'''  ---------------------------------------------------------  '''

def get_users_like_location(unknown_users, local_users, city, boundingbox, infolder, outfolder, users_homes):

    users_likes_locations = {}
    venues_stats          = {}


    print(city + ' --  start parsing likes.json...')



    fout = open(outfolder + '/venues_info/venues_liked_categories_times.dat', 'w')


    for ind, line in enumerate(open(infolder + city + '_likes.json', 'r')):
    


        if ind % 1000 == 0: 
            print (ind)

        #if ind == 100: break

        jsono = json.loads(line)
        user  = jsono['list']['user']['id']

        if str(user) in set(unknown_users + local_users):
    

            for item in  jsono['list']['listItems']['items']:

                venue    = item['venue']
                location = venue['location']
                categ    = 'na'

                venueid  = venue['id']


                if venueid not in venues_stats:
                    venues_stats[venueid] = str(venue['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}

                try:
                    categ = (venue['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                except:
                    pass


                subcateg = 'na'
                try:
                    subcateg = (venue['categories'][0]['name'])#[0]['prefix'])
                except:
                    pass


                createdAt = '0'
                try:
                    createdAt = str(item['createdAt'])
                except:
                    pass

                fout.write(str(user) + '\t' + venueid + '\t' + categ + '\t' + subcateg + '\t' + createdAt + '\n')



                lng = location['lng']
                lat = location['lat']

                venue = (venueid, lng, lat, categ)





               # print (venueid, categ, subcateg)






                if check_box(boundingbox, city, lat, lng):
                    if 'Residential' in str((venue)):

                        if user not in users_homes:
                            users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + venueid])
                        else:
                            users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + venueid))


                if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                    if user not in users_likes_locations:
                        users_likes_locations[user] = [venue]
                    else:
                        users_likes_locations[user].append(venue)
      

    fout.close()

    f = open(outfolder + '/venues_info/' + city + '_liked_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()

    return users_likes_locations





'''  ---------------------------------------------------------  '''
'''  harness the locations of places which from the users piced '''
'''  ---------------------------------------------------------  '''

def get_photos_locations_and_users(unknown_users, local_users, city, boundingbox, infolder, outfolder, users_homes):


    users_photos  = {}
    venues_stats  = {}


    fout = open(outfolder + '/venues_info/venues_photod_categories_times.dat', 'w')

    print(city + ' --  start parsing photos.json...')

    for ind, line in enumerate(open(infolder + city + '_photos.json', 'r')):

        if ind % 1000 == 0: print (ind)

        #if ind == 100: break


        jsono = json.loads(line)

        count = jsono['totalCount']
        user  = jsono['id']


        if str(user) in set(unknown_users + local_users) and count > 0:



            for index, item in enumerate(jsono['photos']['items']):

                if 'venue' in item:

                    venue    = item['venue']
                    venueid  = venue['id']
                    location = venue['location']


                    if venueid not in venues_stats:
                        venues_stats[venueid] = str(venue['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}



                    categ = 'na'
                    try:
                        categ = (venue['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                    except:
                        pass


                    subcateg = 'na'
                    try:
                        subcateg = (venue['categories'][0]['name'])#[0]['prefix'])
                    except:
                        pass


                    createdAt = '0'
                    try:
                        createdAt = str(item['createdAt'])
                    except:
                        pass

                    fout.write(str(user) + '\t' + venueid + '\t' + categ + '\t' + subcateg + '\t' + createdAt + '\n')




                    lng = venue['location']['lng']
                    lat = venue['location']['lat']


                    try:

                        if check_box(boundingbox, city, lat, lng):
                            if 'Residential' in venue['categories'][0]['shortName']:
                            
                               if check_box(boundingbox, city, lat, lng):        
                                
                                   if user not in users_homes:
                                        users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + venueid])
                                   else:
                                        users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + venueid))

                    except: 
                        pass



                    photo = (venueid, lng, lat, categ )  

                    if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                        if user not in users_photos:
                            users_photos[user] = [photo]
                        else:
                            users_photos[user].append(photo)


    fout.close()


    f = open(outfolder + '/venues_info/' + city + '_photoed_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()



    return users_photos















'''  ---------------------------------------------------------  '''
'''  harness the locations of places which suggested as tips    '''
'''  ---------------------------------------------------------  '''

def get_tips_locations_and_users(unknown_users, local_users, city, boundingbox, infolder, outfolder, users_homes):


    users_tips    = {}
    venues_stats  = {}

    print(city + ' --  start parsing tips.json...')



    fout = open(outfolder + '/venues_info/venues_tipped_categories_times.dat', 'w')


    for ind, line in enumerate(open(infolder + city + '_tips.json', 'r')):


        if ind % 1000 == 0: print (ind)

        #if ind == 100: break


        jsono = json.loads(line)
        user  = jsono['id']

        if str(user) in set(unknown_users + local_users):

            for item in jsono['list']['listItems']['items']:

                venue   = item['venue']
                venueid = venue['id']

                lng = venue['location']['lng']
                lat = venue['location']['lat']

                try:
                    if 'Residential' in venue['categories'][0]['shortName']:
                    
                        if check_box(boundingbox, city, lat, lng):
                            
                            if user not in users_homes:
                                users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + venueid])
                            else:
                                users_homes[user].add((str(lng)   + '\t' + str(lat) + '\t' + venueid))

                except: 
                    pass


                if venueid not in venues_stats:
                    venues_stats[venueid] = str(venue['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}


                categ = 'na'
                try:
                    categ = (venue['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                except:
                    pass


                subcateg = 'na'
                try:
                    subcateg = (venue['categories'][0]['name'])#[0]['prefix'])
                except:
                    pass


                createdAt = '0'
                try:
                    createdAt = str(item['createdAt'])
                except:
                    pass

                fout.write(str(user) + '\t' + venueid + '\t' + categ + '\t' + subcateg + '\t' + createdAt + '\n')



                tip = (venueid, lng, lat, categ )  


                if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                    if user not in users_tips:
                        users_tips[user] = [tip]
                    else:
                        users_tips[user].append(tip)


    fout.close()

    f = open(outfolder + '/venues_info/' + city + '_tipped_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()


    return users_tips





'''  ---------------------------------------------------------  '''
'''                 merge venue categories                      '''
'''  ---------------------------------------------------------  '''

def merge_venue_categories_stuff( city, infolder, outfolder):


    files = [   outfolder + '/venues_info/venues_tipped_categories_times.dat',
                outfolder + '/venues_info/venues_photod_categories_times.dat',
                outfolder + '/venues_info/venues_liked_categories_times.dat']


    lines = set()

    for fn in files:
        for line in open(fn):
            lines.add(line)


    lines = list(lines)

    outf =  open(outfolder + '/venues_info/venues_all_categories_times.dat', 'w')
    for line in lines:
        outf.write(line)
    outf.close()



def get_category_stat( city, infolder, outfolder):


    venues_cats    = {}
    venues_subcats = {}

    for line in open(outfolder + '/venues_info/venues_all_categories_times.dat'):
        user, venue, cat, subcat, time = line.strip().split('\t')

        venues_cats[venue]    = cat
        venues_subcats[venue] = subcat



    CNT_cat    = Counter(list(venues_cats.values()))
    CNT_subcat = Counter(list(venues_subcats.values()))

    n_cat    = float(sum(CNT_cat.values()))
    n_subcat = float(sum(CNT_subcat.values()))


    fout = open(outfolder + '/venues_info/venues_category_frequency.dat',    'w')
    gout = open(outfolder + '/venues_info/venues_subcategory_frequency.dat', 'w')


    for item, freq in CNT_cat.items():
        fout.write (item + '\t' + str(freq) + '\t' + str(n_cat) + '\t' + str(freq / n_cat) + '\n')


    for item, freq in CNT_subcat.items():
        gout.write (item + '\t' + str(freq) + '\t' + str(n_cat) + '\t' + str(freq / n_subcat) + '\n')




    fout.close()
    gout.close()
    
   # print (Counter(list(venues_subcats.values())))



'''  ---------------------------------------------------------  '''
'''   list the friends of all the users for the friendship nw   '''
'''  ---------------------------------------------------------  '''

def get_users_friends(local_users, city, infolder, outfolder):

    users_friends = {}

    print(city + ' --  start parsing friends.json...')

    for ind, line in enumerate(open(infolder + city + '_friends.json', 'r')):

        #if ind == 5: break

        jsono   = json.loads(line)      
        user    = jsono['id']

        if user in local_users:

            friends = jsono['friends']['items']    

            if len(friends) > 0:

                for friend in friends:

                    friendid = friend['id']
         
                    if user not in users_friends:
                        users_friends[user] = [friendid]
                    else:
                        users_friends[user].append(friendid)

    fout = open(outfolder + '/user_info/' + city + '_users_friends.dat', 'w')
    fout.write('user\tall_the_friends\n')
    for user, friends in users_friends.items():
        fout.write(str(user) + '\t' + '\t'.join([str(fff) for fff in list(set(friends))]) + '\n')
    fout.close()


    fout = open(outfolder + '/user_info/' + city + '_users_friends.lgl', 'w')
    for user, friends in users_friends.items():
        fout.write('# ' + user + '\n')
        for friend in list(set(friends)):
            fout.write(friend + '\n')
    fout.close()

    return users_friends




'''  ---------------------------------------------------------  '''
'''                     get users home cities                   '''
'''  ---------------------------------------------------------  '''


''' three types of users:
        - live in bristol
        - dont live in bristol
        - we dont know
'''


def get_local_users(city, infolder, outfolder):



    print(city + ' --  get the users homeCities...')

    users_tips = {}
    hres       = open(outfolder + 'user_info/'  + city + '_users_locals_homeCities.dat',    'w')
    gres       = open(outfolder + 'user_info/'  + city + '_users_nonlocals_homeCities.dat', 'w')
    qres       = open(outfolder + 'user_info/'  + city + '_users_all_users.dat', 'w')

    for ind, line in enumerate(open(infolder + city + '_users.json', 'r')):

        jsono = json.loads(line)
        user  = jsono['id']

        qres.write(user + '\n')

        if 'homeCity' in jsono:

            hcity = jsono['homeCity']

            if len(hcity) > 0:
                if city in hcity.lower():
                    hres.write(user + '\t' + hcity + '\n')
                else:        
                    gres.write(user + '\t' + hcity + '\n')

    hres.close()
    gres.close()
    qres.close()


    localss   = set([line.strip().split('\t')[0] for line in open(outfolder + 'user_info/'  + city + '_users_locals_homeCities.dat')])
    nonlocals = set([line.strip().split('\t')[0] for line in open(outfolder + 'user_info/'  + city + '_users_nonlocals_homeCities.dat')])
    everyone  = set([line.strip().split('\t')[0] for line in open(outfolder + 'user_info/'  + city + '_users_all_users.dat')])

    unknown   = list((everyone.difference(localss)).difference(nonlocals))
    pres      = open(outfolder + 'user_info/'  + city + '_users_unknown_homeCities.dat',    'w')


    for u in unknown:
        pres.write(u + '\n')
    pres.close()


    fout = open(outfolder + 'basic_stats/user_types_count.dat', 'w')

    fout.write('Total number of users: '                          +  str(len(localss) + len(nonlocals) +  len(unknown)      )   + '\n')
    fout.write('Number of local users (homeCity here): '          +  str(len(localss))   + '\n')
    fout.write('Number of non-local users (homeCity not here): '  +  str(len(nonlocals)) + '\n')
    fout.write('Number of unknown users (no homeCity given): '    +  str(len(unknown))   + '\n')

    fout.close()

    return list(unknown), list(localss), list(nonlocals)




'''  ---------------------------------------------------------  '''
'''     write out the home locations as groundtruth             '''
'''  ---------------------------------------------------------  '''

def write_home_locations(users_homes, city, outfolder, users):

    fres = open(outfolder + 'user_info/'  + city + '_groundtruth_home_locations_all.dat',    'w')
    gres = open(outfolder + 'user_info/'  + city + '_groundtruth_home_locations_unique.dat', 'w')

    print(city + ' --  write home locations...')

    num_of_home = []

    for user, home in users_homes.items():

        fres.write(str(user) + '\t' + '\t'.join(list(home)) + '\n')

        if len(home) == 1:
            gres.write(str(user) + '\t' + '\t'.join(list(home)) + '\n')
    
        num_of_home.append(len(home))
       

    plt.hist(num_of_home)
    plt.title(str(len(users_homes)) + '/' + str(users) + '  have home locations') 
    plt.savefig(outfolder + 'figures/' + city + '_home_locations_freq.png')
    plt.close()

    fres.close()
    gres.close()
 
    ffout = open(outfolder + 'basic_stats/user_types_count.dat', 'a')
    ffout.write('Number of users w unique local Residence coordinates: ' + str(len([line.strip() for line in open(outfolder + 'user_info/'  + city + '_groundtruth_home_locations_unique.dat') ])) + '\n')
    ffout.close()


def get_venues_stat(city, outfolder):


    fout = open(outfolder + 'basic_stats/user_types_count.dat', 'a')


    files = [outfolder + '/venues_info/' + fn for fn in [city + '_liked_venues_stats.dat', city + '_photoed_venues_stats.dat', city + '_tipped_venues_stats.dat']]

    n = []
    for fn in files:
        n += [line.strip().split('\t')[0] for line in open(fn)]

    print (len(set(n)))

    fout.write('\nTotal number of venues: ' +  str(len(set(n)))  + '\n')
    fout.close()
    




'''  ---------------------------------------------------------  '''
'''  get the venues of the users merging liked and photod       '''
'''  ---------------------------------------------------------  '''

def get_users_venues(unknown_users, local_users, users_photos, users_likes, users_tips, city, outfolder):


    print(city + ' --  get a users venues...')

    users_venues = {}

    for user, photos in users_photos.items():

        if user not in users_venues:
            users_venues[user] = [p[0] for p in photos]
        else:
            users_venues[user] += [p[0] for p in photos]


    for user, likes in users_likes.items():

        if user not in users_venues:
            users_venues[user] = [l[0] for l in likes]
        else:
            users_venues[user] += [l[0] for l in likes]


    for user, likes in users_tips.items():

        if user not in users_venues:
            users_venues[user] = [l[0] for l in likes]
        else:
            users_venues[user] += [l[0] for l in likes]


    fres = open(outfolder + '/user_info/' + city + '_users_venues.dat', 'w') 
    for user, venues in users_venues.items():
        fres.write(str(user) + '\t' + '\t'.join(list(set(venues))) + '\n')
    fres.close()
    



'''  ---------------------------------------------------------  '''
'''  get the basic information about the venues                 '''
'''  ---------------------------------------------------------  '''

def get_venues_information(city, boundingbox, infolder, outfolder):

    venues_stats = {}
    venues_coord = {}       

    print(city + ' --  start parsing venues.json...')

    for line in open(infolder + city + '_venues.json', 'r'):

        jsono = json.loads(line)
        venue = jsono['id']
        stats = jsono['stats']
        lng   = jsono['location']['lng']
        lat   = jsono['location']['lat']

        if check_box(boundingbox, city, lat, lng):

            venues_stats[venue] = stats
            venues_coord[venue] = str(lng) + '\t' +  str(lat)


    df = pd.DataFrame.from_dict(venues_stats, orient = 'index')
    df.to_csv(outfolder + '/venues_info/' + city + '_venues_success_stats.dat', sep = '\t')


    f = open(outfolder + '/venues_info/' + city + '_venues_locations.dat', 'w')
    for venue, coord in venues_coord.items():
        f.write(venue + '\t' + coord + '\n')
    f.close()




'''  ---------------------------------------------------------  '''
'''  calc the distane mtx between venues for further stuff      '''
'''  ---------------------------------------------------------  '''


def venues_distance_mtx(boundingbox, city, outfolder):


    print(city + ' --  get venues distance matrix...')

    fout = open(outfolder + '/venues_info/' + city + '_venues_distance_matrix.dat', 'w')

    for line in open(outfolder + '/user_info/' + city + '_user_venues_full.dat'):

        venues = [vvv.split(',') for vvv in (line.strip().split('\t')[1:])]

        for v1 in venues:

            id1  = v1[0]
            lng1 = float(v1[1])
            lat1 = float(v1[2])
    
            if check_box(boundingbox, city, lat1, lng1):

                for v2 in venues:

                    if v1 != v2:

                        id2 = v2[0]
                        lng2 = float(v2[1])
                        lat2 = float(v2[2])

                        if check_box(boundingbox, city, lat2, lng2):

                            fout.write ( id1 + '\t' +  id2 + '\t' + str(mpu.haversine_distance((lat1, lng1), (lat2, lng2))) + '\n')

    fout.close()
        



'''  ---------------------------------------------------------  '''
'''  merge and write out all the locations of the users         '''
'''  ---------------------------------------------------------  '''


#check_box(boundingbox, city, lat, lng)


def get_users_coordinates(users_homes, local_users, unknown_users, users_likes, users_tips, users_photos, city, outfolder, bbox):

    print(city + ' --  get users coordinates...')

    # merge users' locations


    likes_tips_photos = {}

    for u, v in users_likes.items():
        u = str(u)
        if u not in likes_tips_photos:
            likes_tips_photos[u] = v
        else:
            likes_tips_photos[u] += v


    for u, v in users_tips.items():
        u = str(u)
        if u not in likes_tips_photos:
            likes_tips_photos[u] = v
        else:
            likes_tips_photos[u] += v


    for u, v in users_photos.items():
        u = str(u)
        if u not in likes_tips_photos:
            likes_tips_photos[u] = v
        else:
            likes_tips_photos[u] += v



    

    ''' save coordinates '''
    f = open(outfolder + '/user_info/' + city + '_user_coordinates_raw.dat', 'w')
    g = open(outfolder + '/user_info/' + city + '_user_venues_full.dat',     'w')
    h = open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat',     'w')
    l = open(outfolder + '/user_info/' + city + '_user_coordinates_raw_locals_filtered.dat', 'w')
    f.write('userid\tcoordinates\n')


    for user, venues in likes_tips_photos.items():


        f.write(str(user) + '\t')            
        g.write(str(user) + '\t')
        h.write(str(user) + '\t')            
        l.write(str(user) + '\t')



        venues = list(set(venues))


        f.write('\t'.join([ str(v[1]) + ', ' + str(v[2]) for v in venues] )  + '\n')
        g.write('\t'.join([v[0] + ',' + str(v[1]) + ', ' + str(v[2]) +  ',' + v[3] for v in venues] )  + '\n')



        if int(user) in users_homes:

            for venue in venues:
    
                if check_box(bbox, city, venue[2], venue[1]):
     
                    l.write( str( venue[1] ) + ', ' + str( venue[2] ) + '\t' )
                    h.write(venue[0] + ',' + str( venue[1] ) + ', ' + str( venue[2] ) +  ',' + venue[3] + '\t')

        else:
            for venue in venues:
                l.write( str( venue[1] ) + ', ' + str( venue[2] ) + '\t' )
                h.write(venue[0] + ',' + str( venue[1] ) + ', ' + str( venue[2] ) +  ',' + venue[3] + '\t')

        l.write('\n')
        h.write('\n')


    f.close()
    g.close()
    h.close()   
    l.close()



'''  ---------------------------------------------------------  '''
'''   calc the distance between the users home and other locs   '''
'''  ---------------------------------------------------------  '''

def get_users_distance_distr_from_home(city, outfolder):

    print(city + ' --  distance of users locations and home...')

    users_home   = {}
    users_venues = {}
    user_dist    = {}

    for line in open( outfolder + 'user_info/'  + city + '_groundtruth_home_locations_unique.dat' ):
        user, lng, lat, venue =  line.strip().split('\t')
        lng, lat = float(lng), float(lat)
        users_home[user] = (lng, lat)


    for line in open( outfolder + '/user_info/' + city + '_user_venues_full.dat'):
        fields = (line.strip().split('\t'))
        user   = fields[0]

        if user in users_home and user not in user_dist:

            venues = [(float(vv.split(',')[1]), float(vv.split(',')[2]), vv.split(',')[0]) for vv in fields[1:]]

            for (lngv, latv, venue) in venues:
                user_dist[user] = mpu.haversine_distance((latv, lngv), (users_home[user][1], users_home[user][0]))



    users_num_homes = []
    for ind, line in enumerate(open( outfolder + 'user_info/'  + city + '_user_venues_full_locals_filtered.dat')):
        #if ind == 100: break
    
        users_num_homes.append(len(line.strip().split('\t')))# (len(line.strip().split('\t')[1:])/3.0)







    f, ax = plt.subplots(1, 3, figsize=(18, 5))

    
    ax[0].hist(users_num_homes, bins = 60)
    ax[0].set_xlabel('Users\'s number of venues', fontsize = 12)
    ax[0].set_yscale('log')



    ax[1].hist([d for d in users_num_homes if d < 50], bins = 20)
    ax[1].set_xlabel('Users\'s number of venues', fontsize = 12)
    ax[1].set_yscale('log')




    ax[2].hist([ d for d in   list(user_dist.values())  if d > 0.0 and d < 10.0]   , bins = 60, alpha = 0.8)
    ax[2].set_xlabel('Users\'s locations\' distances from their home location [km]', fontsize = 12)


    plt.show()

    plt.savefig(outfolder + 'figures/' + city + '_distances_from_home_locations.png')
    print('Figure saved.')
    plt.close()









'''  ---------------------------------------------------------  '''
'''             get users similarity matrix                     '''
'''  ---------------------------------------------------------  '''

def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_users_similarity_mtx(city, outfolder):

    print(city + ' --  get users similarity matrix...')

    users_venues = {} 
    fout         = open(outfolder + '/user_info/' + city + '_users_jaccard_sim_mtx.dat', 'w')

    for line in open( outfolder + '/user_info/' + city + '_user_venues_full.dat'):
        fields                  = (line.strip().split('\t'))
        users_venues[fields[0]] = set([ vv.split(',')[0] for vv in fields[1:]])


    for u1, venues1 in users_venues.items():
        for u2, venues2 in users_venues.items():
            fout.write( u1 + '\t' + u2 + '\t' + str( jaccard(venues1, venues2)) + '\n')

    fout.close()




'''  ---------------------------------------------------------  '''
'''                get venuesusers                              '''
'''  ---------------------------------------------------------  '''

def get_venues_users(city, outfolder):

    print(city + ' --  get venues of users...')

    venues_users = {}

    for line in open( outfolder + '/user_info/' + city + '_user_venues_full.dat'):
        fields  = (line.strip().split('\t'))
        user    = fields[0]
        venues  = [ vv.split(',')[0] for vv in fields[1:]]
 
        for v in venues:
            if v not in venues_users:
                venues_users[v] = [user]
            else:
                venues_users[v].append(user)

    fout = open(outfolder + '/venues_info/' + city + '_venues_users.dat', 'w')
    for v, users in venues_users.items():
        fout.write(v + '\t' + '\t'.join(users) + '\n')
    fout.close()



'''  ---------------------------------------------------------  '''
'''  just a helper to create new fodlers                        '''
'''  ---------------------------------------------------------  '''

def create_folder(newfolder):

   if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    



