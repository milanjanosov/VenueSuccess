import pandas as pd
import geopandas as gpd
import numpy as np
import time 
import os
from ParseJsons import check_box
from multiprocessing import Process
from geopandas.geoseries import Point


''' =========================================================== '''
'''             load the shapefile of UK                        '''
''' =========================================================== '''


def load_shp(city):
    
    t1 = time.time()

    print ('Loading the WARD shapefile...')
    ward_shp_df = gpd.read_file('wards/London-wards-2014_ESRI/London_Ward.shp').to_crs({'init': 'epsg:4326'})  
    print ('Shapefile loaded\t', time.time() - t1)
    return ward_shp_df





''' =========================================================== '''
'''          parse the coordinates of the 4sqr venues           '''
''' =========================================================== '''

def get_venues_coordinates(city, outfolder):

    t1 = time.time()
    print ('Parsing venue coordinates...')

    venues_coordinates = {}

    for ind, line in enumerate(open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat')):

        #if ind == 5000: break
        if ind % 5000 == 0:
            print (ind)
        fields = line.strip().split('\t')
        venues = [fff.split(',') for fff in fields[1:]]

        for v in venues:
            venues_coordinates[v[0]] = (float(v[1]), float(v[2]))


    print ('Venue coordinates parsed\t', time.time() - t1)
    return venues_coordinates





''' =========================================================== '''
'''              get the venues within ward-s                   '''
''' =========================================================== '''


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        



def get_wards_paralel(args):
    

    venues_coord_chunks = args[0]
    cityshape           = args[1]
    thread_id           = args[2]
    bbox                = args[3]
    city                = args[4]
    outfolder           = args[5]
    nnn                 = len(venues_coord_chunks)


    fout = open(outfolder + '/venue_ward_attributes_' + str(thread_id), 'w')

    for ind, (venue, coord) in enumerate(venues_coord_chunks.items()):

        if ind % 100 == 0: 
            print (thread_id, '\t', ind, '/', nnn)

        lat = float(coord[1])
        lng = float(coord[0])

        if check_box(bbox, city, lat, lng):

            pnt      = Point(lng, lat)        
            query_df = cityshape[cityshape.contains(pnt)]
            if query_df.shape[0] == 1:


                try:
                    ward, polygon = (query_df.iloc[0]['GSS_CODE'], query_df.iloc[0]['geometry'])

                    bounds  = polygon.bounds
                    lng0    = str(bounds[0])
                    lat0    = str(bounds[1])
                    lng1    = str(bounds[2])
                    lat1    = str(bounds[3])
                    length  = str(polygon.length)
                    area    = str(polygon.area)

                    fout.write ('\t'.join([venue, str(lng), str(lat), ward, lng0, lat0, lng1, lat1, length, area]) + '\n')

                except:
                    pass

    fout.close()





def get_ward_venues(cityshape, venues_coordinates, bbox, city, outfolder_):

    t1 = time.time()
    print ('Converting (lat, long) to WARDS-s...')
    
    ward_venues   = {}
    ward_polygons = {}



    num_threads  = 40
    venues       = list(venues_coordinates.keys())
    venue_chunks = chunkIt(venues, num_threads)



    outfolder = outfolder_ + '/venues_info/ward_venues_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)



    Pros = [] 
    for i in range(0,num_threads):  
        venues_coord_chunks = {k : venues_coordinates[k] for k in venue_chunks[i] }
        p = Process(target = get_wards_paralel, args=([venues_coord_chunks, cityshape, i, bbox, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    wards = set()
    files = os.listdir(outfolder)
    fout  = open(outfolder_  + '/venues_info/venues_ward_full.dat', 'w')
    fout.write('venue\tlng\tlat\ward\tlng0\tlat0\tlng1\tlat1\tlength\tarea\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            wards.add(line.strip().split('\t')[3])
            fout.write(line)
    fout.close()
    print('Number of WARDS: ', len(wards))

    return ward_venues, ward_polygons



''' =========================================================== '''
'''         parse the edges of the venue sim nw                 '''
''' =========================================================== '''





def get_edge_weights2(city, outfolder, venues_users, ward_venues):
    

    t1 = time.time()
    edges_weights2 = {}
    print ('Parsing venue similarity network edge list...')
    
    nnn = len(ward_venues)    

    for ind, (ward, venues) in enumerate(ward_venues.items()):
                          
        print(ind, '/', nnn)
               
#        if ind == 100: break
                                         
        for v1 in venues:
            for v2 in venues:
                if v1 != v2:
                    edge   = '_'.join(sorted([v1, v2]))
                    w = len(set(venues_users[v1]).intersection(set(venues_users[v2])))
                    if w > 0:
                        edges_weights2[edge] = w             

    print ('Venues similarity network edges parsed\t', time.time() - t1)
    return edges_weights2
        
    

    


''' =========================================================== '''
'''     get users stuff liking venues within the same ward      '''
''' =========================================================== '''
    

def get_users_friends(outfolder, city):
    
    friends_list = {}

    t1 = time.time()
    print ('Get users friends')
        

    for ind, line in enumerate(open( outfolder + 'networks/gephi/' + city + '_friendship_edges.dat')):
 
        if 'Source' not in line:
        
#            if ind == 100: break
            
            source, target, a, b, c = line.strip().split('\t')
            
            if source not in friends_list:
                friends_list[source] = [target]
            else:
                friends_list[source].append(target)
            
        
            if target not in friends_list:
                friends_list[target] = [source]
            else:
                friends_list[target].append(source)

    print ('Users friends parsed\t', time.time() - t1)
    return friends_list
        



def get_friendship_ties_within_ward(ward_venues, venues_users, friends_list):
    
    t1 = time.time()
    print ('Get friendship ties within ward')

    ward_friendships = {}
    ward_num_users   = {}
    
    nnn = len(ward_venues)

    for ind, (ward, venues) in enumerate(ward_venues.items()):
        
#        if ind == 100 : break
 #       print (ind, '/', nnn)

        users = []
        for venue in venues:
            users += venues_users[venue]
    
    
        ward_num_users[ward] = len(users)
    
        for u1 in users:
            if u1 in friends_list:
                for u2 in users:
                    if u1 != u2:
                        if u2 in friends_list:
                            edge = '_'.join(sorted([u1, u2]))
    
                            if ward not in ward_friendships:
                                ward_friendships[ward] = [edge]
                            else:
                                ward_friendships[ward].append(edge)
    
    print('ward friendship ties processed\t', time.time() - t1)    

    return ward_num_users, ward_friendships    




''' =========================================================== '''
'''        get users sutff living in the same ward              '''
''' =========================================================== '''






#####################################################################################



def get_user_wards_paralel(args):
    

    user_coord_chunks = args[0]
    cityshape         = args[1]
    thread_id         = args[2]
    city              = args[3]
    outfolder         = args[4]
    nnn               = len(user_coord_chunks)


    fout = open(outfolder + '/venue_ward_attributes_' + str(thread_id), 'w')

    for ind, (user, coord) in enumerate(user_coord_chunks.items()):

        #if ind == 50: break
        if ind % 100 == 0: 
            print (thread_id, '\t', ind, '/', nnn)

        lat = float(coord[1])
        lng = float(coord[0])


        pnt      = Point(lng, lat)        
        query_df = cityshape[cityshape.contains(pnt)]

        if query_df.shape[0] == 1:
            ward = query_df.iloc[0]['GSS_CODE']
            fout.write (user + '\t' + ward + '\n')

    fout.close()



def get_users_ward(city, outroot, cityshape):
    
  
    print ('Get users ward...')

    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'

    users_homes = {}
    for ind, line in enumerate(open(infile)):
        user, lng, lat    = line.strip().split('\t')
        users_homes[user] = (float(lng), float(lat))



    num_threads  = 40
    users        = list(users_homes.keys())
    users_chunks = chunkIt(users, num_threads)



    outfolder = outroot + '/user_info/ward_user_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)




    Pros = [] 
    for i in range(0,num_threads):  
        user_coord_chunks = {k : users_homes[k] for k in users_chunks[i] }
        p = Process(target = get_user_wards_paralel, args=([user_coord_chunks, cityshape, i, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    files = os.listdir(outfolder)
    fout  = open(outroot  + '/user_info/user_wards.dat', 'w')
    fout.write('user\tward\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            fout.write(line)
    fout.close()
            






def friendships_within_ward(ward_users, friends_list):
                

    t1 = time.time()
    print ('Get friendships within ward')

    users_ward = {}
    for ward, users in ward_users.items():
        for user in users:
            users_ward[user] = ward

        
    ward_friendships = {}    
        
        
    for user, friends in friends_list.items():

        
        if user in users_ward:

            user_ward = users_ward[user]

            if str(user_ward) != 0:

                for friend in friends:
                    if user_ward == users_ward[friend]:

                        friendship = '_'.join([user, friend])

                        if ward not in ward_friendships:
                            ward_friendships[user_ward] = set([friendship])
                        else:
                            ward_friendships[user_ward].add(friendship)

    

                    
        
    for ward in ward_users:
        if ward not in ward_friendships:
            ward_friendships[ward] = 0
        else:
            ward_friendships[ward] = len(ward_friendships[ward])
    
    print ('ward USERS: ', len(ward_users), '\t', time.time() - t1)
        
    return ward_friendships
        

    
    
''' =========================================================== '''
'''      get the network properties of the ward level nws       '''
''' =========================================================== '''

def get_ward_mininw_edges(ward_venues, edges_weights):
    
    t1 = time.time()
    print ('Create the ward level mini-network edge lists...')

    ward_edges = {}

    for ind, (ward, venues) in enumerate(ward_venues.items()):
        #if ind == 10: break

        for v1 in venues:
            for v2 in venues:
                if v1 != v2:
                    edge = '_'.join(sorted([v1,v2]))

                    if edge in edges_weights:

                        weight = edges_weights[edge]    

                        if ward not in ward_edges:
                            ward_edges[ward] = [(edge, weight)]
                        else:
                            ward_edges[ward].append((edge, weight))

    print ('ward level networks..\t', time.time() - t1)

    return ward_edges




''' =========================================================== '''
'''            get the stuff on the ward network level          '''
''' =========================================================== '''

def get_ward_nw_features(ward_venues, ward_edges, edges_weights):

    t1 = time.time()
    print ('Deriving the ward network features...')

    all_edges = [set(e.split('_')) for e in edges_weights.keys()]
    all_nodes = list(([ e.split('_')[0]  for e in edges_weights.keys()] + [ e.split('_')[1]  for e in edges_weights.keys()]))
    NNN       = len(all_nodes)

    edge_avg_weight_glb = np.mean(list(edges_weights.values()))                # global avg weight
    edge_density_glb    = len(all_edges) / ( NNN * (NNN - 1) / 2.0 )     # number of existing edges out 

    ward_weights_density = {}
    
    for ind, (ward, venues) in enumerate(ward_venues.items()):

        #if ind == 100: break

        # edge types
        #   - within the ward
        #   - on the boundary (one node wihtin, other outside)
        #   - outside of the ward (both nodes outside)
        # that didnt work out (computationally)  --> comparing ward level stuff to the global stuff

        edge_num               = 0.0            # number of edges within the ward
        node_num               = len(venues)    # number of nodes (venues) within the ward
        edge_avg_weight_in     = 0.0            # avg edge weight within the ward
        edge_density_in        = 0.0            # number of existing edges within ward out of possible ones


        venue_num = len(venues)

        if venue_num > 1 and ward in ward_edges:

            weights  = [e[1] for e in ward_edges[ward]]
            edges    = [e[0] for e in ward_edges[ward]]   
            nodes    = venues

            edge_num            = len(weights)    
            edge_avg_weight_in  = sum(weights) / edge_num 
            edge_density_in     = edge_num / ( venue_num * (venue_num - 1 ) / 2.0  )

            ward_weights_density[ward] = (edge_avg_weight_in, edge_density_in)
            

    print ('ward level network features\t', time.time() - t1)

    return ward_weights_density, edge_density_glb, edge_avg_weight_glb



def get_venues_features(ward_polygons,ward_local_friendships, ward_users, ward_venues, ward_num_users, ward_friendships, ward_weights_density, edge_density_glb, edge_avg_weight_glb, outfolder, city):

    venues_features = {}

    for ind, (ward, venues) in enumerate(ward_venues.items()):
 
        polygon     = ward_polygons[ward]
        bounds      = polygon.bounds
        lng0        = bounds[0]
        lat0        = bounds[1]
        lng1        = bounds[2]
        lat1        = bounds[3]
        length      = polygon.length
        area        = polygon.area
        usersnum    = 0
        friendships = 0
        livingthere = 0
        wardfriends = 0
        
        if ward in ward_weights_density:
            ward_weight = ward_weights_density[ward][0]
            ward_dens   = ward_weights_density[ward][1]
        else:
            ward_weight = 0.0
            ward_dens   = 0.0
            
        if ward in ward_users:
            livingthere = len(ward_users[ward])
            
            
        if ward in ward_num_users:
            usersnum = ward_num_users[ward]
            
        if ward in ward_friendships:
            friendships = len(ward_friendships)
            
        if ward in ward_local_friendships:
            wardfriends = ward_local_friendships[ward]
  
        if area < 0.01:

            for venue in venues:

                feats = { 'bnds_lng0'       : bounds[0],
                          'bnds_lat0'       : bounds[1],
                          'bnds_lng1'       : bounds[2],
                          'bnds_lat1'       : bounds[3],
                          'bnds_length'     : polygon.length,
                          'bnds_area'       : polygon.area,
                          'ward_weight'     : ward_weight,
                          'ward_dens'       : ward_dens,
                          'ward_rel_weight' : ward_weight / edge_avg_weight_glb,
                          'ward_rel_dens'   : ward_dens   / edge_density_glb,
                          'userslikingnum'  : usersnum,
                          'friendships'     : friendships,
                          'livingthere'     : livingthere,
                          'wardfriends'     : wardfriends
                        }

                venues_features[venue] = feats

            

    
    filename = outfolder + 'networks/' + city  + '_ward_networkmeasures.csv'  
    df = pd.DataFrame.from_dict(venues_features, orient = 'index')
    df.to_csv(filename, sep = '\t')
    
    return df
    






import pandas as pd
import geopandas as gpd
import numpy as np
import time 
import os
from ParseJsons import check_box
from multiprocessing import Process
from geopandas.geoseries import Point




def read_venues_ward(outfolder, city):

    ward_venues = {}
    infile      = outfolder + '/venues_info/venues_ward_full.dat'

    for line in open(infile):
        if 'ward' not in line:
            fields = line.strip().split('\t')
            venue, ward = fields[0],fields[3]

            if ward not in ward_venues:
                ward_venues[ward] = [venue]
            else:
                ward_venues[ward].append(venue)

    return ward_venues



def read_users_ward(outfolder, city):

    ward_users = {}
    infile     = outfolder + '/user_info/user_wards.dat'

    for line in open(infile):
        if 'ward' not in line:
            user, ward = line.strip().split('\t')
          
            if ward not in ward_users:
                ward_users[ward] = [user]
            else:
                ward_users[ward].append(user)

    return ward_users


def get_venues_users(outfolder, city):
    
    t1 = time.time()
    print ('Getting venues user list...')
    venues_users = {}
    
    for ind, line in enumerate(open(outfolder + '/venues_info/' + city + '_venues_users.dat')):
        
        #if ind == 10: break
        fields = line.strip().split('\t')
        venue  = fields[0]
        users  = fields[1:]
        
        venues_users[venue] = users
        

    print ('Venues user lists parsed\t', time.time() - t1)
    return venues_users
 
    




















''' =========================================================== '''
'''             get and save venues and users ward              '''
''' =========================================================== '''


def ward_preproc(city, outfolder, bbox):
    

    cityshape                   = load_shp(city)
    venues_coordinates          = get_venues_coordinates(city, outfolder)
    ward_venues, ward_polygons  = get_ward_venues(cityshape, venues_coordinates, bbox, city, outfolder)
  #  ward_users                  = get_users_ward(city, outfolder, cityshape)    


''' =========================================================== '''
'''                get ward level network stuff                 '''
''' =========================================================== '''



def get_ward_level_networks( city, outfolder, bbox ):

 
  
#    ward_venues   = read_venues_ward(outfolder, city)
#    ward_users    = read_users_ward(outfolder, city)
#    venues_users  = get_venues_users(outfolder, city)
#    edges_weights = get_edge_weights2(city, outfolder, venues_users, ward_venues)    # node1_node2 -> weight




    nodes_edge_weights     = get_node_edge_list(edges_weights)     # node0 -> [(node1, w1), (node2, w2), ...]


    ward_edges   = get_ward_mininw_edges(ward_venues, edges_weights  )    # edges within the mini ward level networks
    venues_users = get_venues_users(outfolder, city)   
    friends_list = get_users_friends(outfolder, city)
   

    ward_local_friendships = friendships_within_ward(ward_users, friends_list)


    
    '''

    ward_weights_density, edge_density_glb, edge_avg_weight_glb  =  get_ward_nw_features(ward_venues, ward_edges, edges_weights)
    ward_num_users, ward_friendships                             = get_friendship_ties_within_ward(ward_venues, venues_users, friends_list)


    get_venues_features(ward_polygons, ward_local_friendships, ward_users, ward_venues, ward_num_users, ward_friendships, ward_weights_density, edge_density_glb, edge_avg_weight_glb, outfolder, city)      
    '''
   



