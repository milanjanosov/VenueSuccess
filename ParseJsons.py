# https://developer.foursquare.com/docs/api/photos/details 
import os, sys
import json
import numpy as np
import csv
import mpu
import gzip
import pandas as pd
import matplotlib.pyplot as plt




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

    for ind, line in enumerate(open(infolder + city + '_likes.json', 'r')):
    

        #if ind == 5: break

        jsono = json.loads(line)
        user  = jsono['list']['user']['id']

        if str(user) in set(unknown_users + local_users):
    


            for item in  jsono['list']['listItems']['items']:

                location = item['venue']['location']
                categ    = 'na'

                if item['venue']['id'] not in venues_stats:
                    venues_stats[item['venue']['id']] = str(item['venue']['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}

                try:
                    categ = (item['venue']['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                except:
                    pass


                lng = location['lng']
                lat = location['lat']

                venue = (item['venue']['id'], lng, lat, categ)


                if check_box(boundingbox, city, lat, lng):
                    if 'Residential' in str((item['venue'])):

                        if user not in users_homes:
                            users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']])
                        else:
                            users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']))


                if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                    if user not in users_likes_locations:
                        users_likes_locations[user] = [venue]
                    else:
                        users_likes_locations[user].append(venue)
      

    f = open(outfolder + '/venues_info/' + city + '_liked_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()

    return users_likes_locations




'''  ---------------------------------------------------------  '''
'''  harness the locations of places which suggested as tips    '''
'''  ---------------------------------------------------------  '''

def get_tips_locations_and_users(unknown_users, local_users, city, boundingbox, infolder, outfolder, users_homes):


    users_tips    = {}
    venues_stats  = {}

    print(city + ' --  start parsing tips.json...')

    for ind, line in enumerate(open(infolder + city + '_tips.json', 'r')):


        #if ind == 5: break


        jsono = json.loads(line)
        user  = jsono['id']

        if str(user) in set(unknown_users + local_users):

            for item in jsono['list']['listItems']['items']:

                lng = item['venue']['location']['lng']
                lat = item['venue']['location']['lat']

                try:
                    if 'Residential' in item['venue']['categories'][0]['shortName']:
                    
                        if check_box(boundingbox, city, lat, lng):
                            
                            if user not in users_homes:
                                users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']])
                            else:
                                users_homes[user].add((str(lng)   + '\t' + str(lat) + '\t' + item['venue']['id']))

                except: 
                    pass


                if item['venue']['id'] not in venues_stats:
                    venues_stats[item['venue']['id']] = str(item['venue']['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}


                categ = 'na'
                try:
                    categ = (item['venue']['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                except:
                    pass


                tip = (item['venue']['id'], lng, lat, categ )  


                if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                    if user not in users_tips:
                        users_tips[user] = [tip]
                    else:
                        users_tips[user].append(tip)


    f = open(outfolder + '/venues_info/' + city + '_tipped_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()


    return users_tips




'''  ---------------------------------------------------------  '''
'''  harness the locations of places which from the users piced '''
'''  ---------------------------------------------------------  '''

def get_photos_locations_and_users(unknown_users, local_users, city, boundingbox, infolder, outfolder, users_homes):


    users_photos  = {}
    venues_stats  = {}

    print(city + ' --  start parsing photos.json...')

    for ind, line in enumerate(open(infolder + city + '_photos.json', 'r')):

        #if ind == 5: break

        jsono = json.loads(line)

        count = jsono['totalCount']
        user  = jsono['id']


        if str(user) in set(unknown_users + local_users) and count > 0:



            for index, item in enumerate(jsono['photos']['items']):

                if 'venue' in item:

                    location = item['venue']['location']


                    if item['venue']['id'] not in venues_stats:
                        venues_stats[item['venue']['id']] = str(item['venue']['stats'])   # {'usersCount': 4, 'checkinsCount': 718, 'tipCount': 1}



                    categ = 'na'
                    try:
                        categ = (item['venue']['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                    except:
                        pass


                    lng = item['venue']['location']['lng']
                    lat = item['venue']['location']['lat']


                    try:

                        if check_box(boundingbox, city, lat, lng):
                            if 'Residential' in item['venue']['categories'][0]['shortName']:
                            
                               if check_box(boundingbox, city, lat, lng):        
                                
                                   if user not in users_homes:
                                        users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']])
                                   else:
                                        users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']))

                    except: 
                        pass

                    #if check_box(boundingbox, city, lat, lng):

                    photo = (item['venue']['id'], lng, lat, categ )  

                    if (str(user) in local_users and check_box(boundingbox, city, lat, lng)) or (str(user) in unknown_users):
                
                        if user not in users_photos:
                            users_photos[user] = [photo]
                        else:
                            users_photos[user].append(photo)



    f = open(outfolder + '/venues_info/' + city + '_photoed_venues_stats.dat', 'w')
    for v, stat in venues_stats.items():
        f.write(v + '\t' + stat + '\n')
    f.close()



    return users_photos





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

        #if ind == 5: break

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


def get_users_coordinates(users_homes, local_users, users_likes, users_tips, users_photos, city, outfolder, bbox):

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

  


        if int(user) in users_homes:# or user in local_users:
            for venue in venues:

        
                if check_box(bbox, city, venue[2], venue[1]):
         

                    l.write( str( venue[1] ) + ', ' + str( venue[2] ) + '\t' )
                    h.write(venue[0] + ',' + str( venue[1] ) + ', ' + str( venue[2] ) +  ',' + venue[3] + '\t')

        elif (user not in users_homes and local_users):
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


    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].hist(list(user_dist.values()), bins = 100, alpha = 0.8)
    ax[0].set_xlabel('Users\'s locations distances from their home location [km]')
    ax[1].hist([ccc for ccc in user_dist.values() if ccc < 5], bins = 50, alpha = 0.8)
    ax[1].set_xlabel('Users\'s locations distances from their home location [km]')
    plt.savefig(outfolder + 'figures/' + city + '_distances_from_home_locations.png')
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
    



