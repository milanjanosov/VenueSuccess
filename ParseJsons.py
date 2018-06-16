

'''   this script parses the jsons obtained from the mongodb        '''
'''   and gets the coordinate list and home addresses of the users  '''


# https://developer.foursquare.com/docs/api/photos/details 
import os, sys
import json
from pprint import pprint
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

def get_users_like_location(city, boundingbox, infolder, outfolder, users_homes):

    users_likes_locations = {}
    categories = {}


    for line in open(infolder + city + '_likes.json', 'r'):
    
        jsono = json.loads(line)
        user  = jsono['list']['user']['id']


        for item in  jsono['list']['listItems']['items']:

            location = item['venue']['location']
            categ    = 'na'

            try:
                categ = (item['venue']['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
            except:
                pass

            if categ not in categories:
                categories[categ] = 1
            else:
                categories[categ] += 1

            lng = location['lng']
            lat = location['lat']

            venue = (item['venue']['id'], lng, lat, categ)


            if check_box(boundingbox, city, item['venue']['location']['lat'], item['venue']['location']['lng']):


                if 'Residential' in str((item['venue'])):

                    if user not in users_homes:
                        users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']])
                    else:
                        users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']))


                if user not in users_likes_locations:
                    users_likes_locations[user] = [venue]
                else:
                    users_likes_locations[user].append(venue)


    return users_likes_locations




'''  ---------------------------------------------------------  '''
'''   list the friends of all the users for the friendship nw   '''
'''  ---------------------------------------------------------  '''

def get_users_friends(city, infolder, outfolder):

    users_friends = {}

    for line in open(infolder + city + '_friends.json', 'r'):

        jsono   = json.loads(line)      
        user    = jsono['id']
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
'''  harness the locations of places which from the users piced '''
'''  ---------------------------------------------------------  '''

def get_photos_locations_and_users(city, boundingbox, infolder, outfolder, users_homes):


    users_photos = {}

    for line in open(infolder + city + '_photos.json', 'r'):

        jsono = json.loads(line)

        count = jsono['totalCount']
        user  = jsono['id']

        if count > 0:

            for index, item in enumerate(jsono['photos']['items']):

                if 'venue' in item:

                    location = item['venue']['location']


                    categ = 'na'
                    try:
                        categ = (item['venue']['categories'][0]['icon']['prefix'].split('v2')[1].split('/')[1])#[0]['prefix'])
                    except:
                        pass


                    lng = item['venue']['location']['lng']
                    lat = item['venue']['location']['lat']


                    try:
                        if 'Residential' in item['venue']['categories'][0]['shortName']:
                        
                            if check_box(boundingbox, city, lat, lng):        
                                
                               if user not in users_homes:
                                    users_homes[user] = set([str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']])
                               else:
                                    users_homes[user].add((str(lng) + '\t' + str(lat) + '\t' + item['venue']['id']))

                    except: 
                        pass

                    if check_box(boundingbox, city, lat, lng):

                        photo = (item['venue']['id'], lng, lat, categ )  

                        if user not in users_photos:
                            users_photos[user] = [photo]
                        else:
                            users_photos[user].append(photo)

    return users_photos




'''  ---------------------------------------------------------  '''
'''     write out the home locations as groundtruth             '''
'''  ---------------------------------------------------------  '''

def write_home_locations(users_homes, city, outfolder, users):

    fres = open(outfolder + 'user_info/'  + city + '_groundtruth_home_locations_all.dat',    'w')
    gres = open(outfolder + 'user_info/'  + city + '_groundtruth_home_locations_unique.dat', 'w')

    num_of_home = []

    for user, home in users_homes.items():

        fres.write(str(user) + '\t' + '\t'.join(list(home)) + '\n')

        if len(home) == 1:
            gres.write(str(user) + '\t' + '\t'.join(list(home)) + '\n')
    
        num_of_home.append(len(home))
    
    plt.hist(num_of_home)
    plt.title(str(len(users_homes)) + '/' + str(users) + '  have home locations') 
    plt.savefig(outfolder + 'user_info/' + city + '_home_locations_freq.png')
    plt.close()

    fres.close()
    gres.close()




'''  ---------------------------------------------------------  '''
'''  get the venues of the users merging liked and photod       '''
'''  ---------------------------------------------------------  '''

def get_users_venues(users_photos, users_likes, city, outfolder):

    users_venues = {}

    for user, photos in users_photos.items():
        if user not in users_venues:
            users_venues[user] = [p[0] for p in photos]
        else:
            users_venues[user] += [p[0] for p in photos]

    print(len(users_venues))

    for user, likes in users_likes.items():
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

def venues_distance_mtx(city, outfolder):

    fout = open(outfolder + '/venues_info/' + city + '_venues_distance_matrix.dat', 'w')

    for line in open(outfolder + '/user_info/' + city + '_user_venues_full.dat'):

        venues = [vvv.split(',') for vvv in (line.strip().split('\t')[1:])]

        for v1 in venues:

            id1  = v1[0]
            lng1 = float(v1[1])
            lat1 = float(v1[2])
    
            for v2 in venues:

                if v1 != v2:

                    id2 = v2[0]
                    lng2 = float(v2[1])
                    lat2 = float(v2[2])

                    fout.write ( id1 + '\t' +  id2 + '\t' + str(mpu.haversine_distance((lat1, lng1), (lat2, lng2))) + '\n')

    fout.close()
        



'''  ---------------------------------------------------------  '''
'''  merge and write out all the locations of the users         '''
'''  ---------------------------------------------------------  '''

def get_users_coordinates(users_likes, users_friends, users_photos, city, outfolder):


    users = list( list(users_likes.keys()) + list(users_friends.keys()) + list(users_photos.keys()))


    ''' save the outputs '''
    for user in users:
        if user in users_likes:
            liked_locations  = [v[0] for v in users_likes[user]]


    # merge users' locations
    users = list(set( list(users_likes.keys()) + list(users_photos.keys())))
    users_locations = {}
    for user in users:
        if user in users_likes and user in users_photos:
            users_locations[user] = users_likes[user] + users_photos[user]
        elif user in users_likes:
            users_locations[user] = users_likes[user]
        elif user in users_photos:
            users_locations[user] = users_photos[user]

        
    ''' save coordinates '''
    f = open(outfolder + '/user_info/' + city + '_user_coordinates_raw.dat', 'w')
    g = open(outfolder + '/user_info/' + city + '_user_venues_full.dat',     'w')
    f.write('userid\tcoordinates\n')

    for user, venues in users_locations.items():

        f.write(str(user) + '\t')            
        g.write(str(user) + '\t')

        venues = list(set(venues))

        for venue in venues:
            f.write( str( venue[1] ) + ', ' + str( venue[2] ) + '\t' )
            g.write(venue[0] + ',' + str( venue[1] ) + ', ' + str( venue[2] ) +  ',' + venue[3] + '\t')

        f.write('\n')
        g.write('\n')

    f.close()
    g.close()




'''  ---------------------------------------------------------  '''
'''   calc the distance between the users home and other locs   '''
'''  ---------------------------------------------------------  '''

def get_users_distance_distr_from_home(city, outfolder):

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
    ax[0].hist(user_dist.values(), bins = 100, alpha = 0.8)
    ax[0].set_xlabel('Users\'s locations distances from their home location [km]')
    ax[1].hist([ccc for ccc in user_dist.values() if ccc < 5], bins = 50, alpha = 0.8)
    ax[1].set_xlabel('Users\'s locations distances from their home location [km]')
    plt.savefig(outfolder + 'user_info/' + city + '_distances_from_home_locations.png')
    plt.close()
   



'''  ---------------------------------------------------------  '''
'''  just a helper to create new fodlers                        '''
'''  ---------------------------------------------------------  '''

def create_folder(newfolder):

   if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    



