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

'''def load_shp(city):
    
    print ('Loading the shapefile...')
    lsoa_shp_df = gpd.read_file('full2/england_lsoa_2011_gen.shp')

    if 'london' == city:
        

    return lsoa_shp_df[lsoa_shp_df['name'].str.contains(city.title())].to_crs({'init': 'epsg:4326'})  

'''
def load_shp(city):
    
    t1 = time.time()

    print ('Loading the shapefile...')
    lsoa_shp_df = gpd.read_file('area_matching/lsoa_shp/lsoa.shp')
    #return lsoa_shp_df[lsoa_shp_df['name'].str.contains(city.title())].to_crs({'init': 'epsg:4326'})  
    print ('Shapefile loaded\t', time.time() - t1)
    return lsoa_shp_df





''' ========================================================================== '''
''' get the LSOA id and the polygon of each coordinate (if its within the city '''
''' ========================================================================== '''

'''def coordinates_to_lsoa(lats, lons, cityshape):
    
    poly = (0,0)
    
    try:
        pnt = Point(lons, lats)
        query_df = cityshape[cityshape.contains(pnt)]
        if query_df.shape[0] == 1:
            poly = (query_df.iloc[0]['name'], query_df.iloc[0]['geometry'])
    except Exception as exception:
        pass
    
    return poly


'''


def coordinates_to_lsoa(lats, lons, cityshape):
    
    poly = (0,0)
    
    try:
        pnt = Point(lons, lats)
        query_df = cityshape[cityshape.contains(pnt)]
        if query_df.shape[0] == 1:
            poly = (query_df.iloc[0]['lsoa11cd'], query_df.iloc[0]['geometry'])
    except Exception as exception:
        pass
    
    return poly


''' =========================================================== '''
'''          parse the coordinates of the 4sqr venues           '''
''' =========================================================== '''

'''def get_venues_coordinates(city, outfolder):

    print ('Parsing venue coordinates...')

    venues_coordinates = {}

    for ind, line in enumerate(open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat')):
        #if ind == 10: break
        fields = line.strip().split('\t')
        venues = [fff.split(',') for fff in fields[1:]]

        for v in venues:
            venues_coordinates[v[0]] = (float(v[1]), float(v[2]))

    return venues_coordinates

'''


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
'''              get the venues within lsoa-s                   '''
''' =========================================================== '''

def get_lsoa_venues_old(cityshape, venues_coordinates, bbox, city):

    t1 = time.time()
    print ('Converting (lat, long) to LSOA-s...')
    
    lsoa_venues  = {}
    lsoa_polygons = {}

    nnn = len(venues_coordinates)



    for ind, (v, c) in enumerate(venues_coordinates.items()):


        lat = float(c[1])
        lng = float(c[0])
    

        if check_box(bbox, city, lat, lng):

            lsoa, polygon = coordinates_to_lsoa( lat, lng, cityshape )

            if lsoa != 0:          
                       
                if lsoa not in lsoa_polygons:
                    lsoa_polygons[lsoa] = polygon    
                
                if lsoa not in lsoa_venues:
                    lsoa_venues[lsoa] = [v]
                else:
                    lsoa_venues[lsoa].append(v)

    print ('Coordinates converted to LSOA-s\t', time.time() - t1)

    return lsoa_venues, lsoa_polygons







########################################################################


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        



def get_lsoas_paralel(args):
    

    venues_coord_chunks = args[0]
    cityshape           = args[1]
    thread_id           = args[2]
    bbox                = args[3]
    city                = args[4]
    outfolder           = args[5]
    nnn                 = len(venues_coord_chunks)


    fout = open(outfolder + '/venue_lsoa_attributes_' + str(thread_id), 'w')

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
                    lsoa, polygon = (query_df.iloc[0]['lsoa11cd'], query_df.iloc[0]['geometry'])

                    bounds  = polygon.bounds
                    lng0    = str(bounds[0])
                    lat0    = str(bounds[1])
                    lng1    = str(bounds[2])
                    lat1    = str(bounds[3])
                    length  = str(polygon.length)
                    area    = str(polygon.area)

                    fout.write ('\t'.join([venue, str(lng), str(lat), lsoa, lng0, lat0, lng1, lat1, length, area]) + '\n')

                except:
                    pass

    fout.close()





def get_lsoa_venues(cityshape, venues_coordinates, bbox, city, outfolder_):

    t1 = time.time()
    print ('Converting (lat, long) to LSOA-s...')
    
    lsoa_venues   = {}
    lsoa_polygons = {}



    num_threads  = 40
    venues       = list(venues_coordinates.keys())
    venue_chunks = chunkIt(venues, num_threads)



    outfolder = outfolder_ + '/venues_info/lsoa_venues_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)



    Pros = [] 
    for i in range(0,num_threads):  
        venues_coord_chunks = {k : venues_coordinates[k] for k in venue_chunks[i] }
        p = Process(target = get_lsoas_paralel, args=([venues_coord_chunks, cityshape, i, bbox, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    files = os.listdir(outfolder)
    fout  = open(outfolder_  + '/venues_info/venues_lsoa_full.dat', 'w')
    fout.write('venue\tlng\tlat\tlsoa\tlng0\tlat0\tlng1\tlat1\tlength\tarea\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            fout.write(line)
    fout.close()
            

    return lsoa_venues, lsoa_polygons







''' =========================================================== '''
'''         parse the edges of the venue sim nw                 '''
''' =========================================================== '''


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
 
    


def get_edge_weights2(city, outfolder, venues_users, lsoa_venues):
    

    t1 = time.time()
    edges_weights2 = {}
    print ('Parsing venue similarity network edge list...')
    
    
    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):
                                         
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
'''        get the lists of nghbnodes fir each node             '''
''' =========================================================== '''

def get_node_edge_list(edges_weights):

    t1 = time.time()
    print ('Listing each nodes neighbours and those edge weights...')


    # for each node list the edges (weights) in which they are present
    nodes_edge_weights = {}
    nnn = len(edges_weights)    

    for ind, (e, w) in enumerate(edges_weights.items()):
    
#        if ind == 100: break
 #       print (ind, '/', nnn)           
 
        e1, e2 = e.split('_')
        
        if e1 not in nodes_edge_weights:
            nodes_edge_weights[e1] = [(e2, w)]
        else:
            nodes_edge_weights[e1].append((e2, w))

                    
        if e2 not in nodes_edge_weights:
            nodes_edge_weights[e2] = [(e1, w)]
        else:
            nodes_edge_weights[e2].append((e1, w))
        
    print ('Neighbour list created\t', time.time() - t1)
    return nodes_edge_weights
    
    


''' =========================================================== '''
'''     get users stuff liking venues within the same lsoa      '''
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
        



def get_friendship_ties_within_lsoa(lsoa_venues, venues_users, friends_list):
    
    t1 = time.time()
    print ('Get friendship ties within lsoa')

    lsoa_friendships = {}
    lsoa_num_users   = {}
    
    nnn = len(lsoa_venues)

    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):
        
#        if ind == 100 : break
 #       print (ind, '/', nnn)

        users = []
        for venue in venues:
            users += venues_users[venue]
    
    
        lsoa_num_users[lsoa] = len(users)
    
        for u1 in users:
            if u1 in friends_list:
                for u2 in users:
                    if u1 != u2:
                        if u2 in friends_list:
                            edge = '_'.join(sorted([u1, u2]))
    
                            if lsoa not in lsoa_friendships:
                                lsoa_friendships[lsoa] = [edge]
                            else:
                                lsoa_friendships[lsoa].append(edge)
    
    print('LSOA friendship ties processed\t', time.time() - t1)    

    return lsoa_num_users, lsoa_friendships    




''' =========================================================== '''
'''        get users sutff living in the same lsoa              '''
''' =========================================================== '''


def get_users_lsoa_old(city, outroot, cityshape):
    
    
    t1 = time.time()
    print ('Get users lsoa...')

    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'

    lsoa_users = {}



    for ind, line in enumerate(open(infile)):
        user, lng, lat = line.strip().split('\t')


#        print (ind)
#        if ind == 100: break
        
        lsoa, polygon = coordinates_to_lsoa( float(lat), float(lng), cityshape ) 
        
        if lsoa not in lsoa_users:
            lsoa_users[lsoa] = [user]
        else:
            lsoa_users[lsoa].append(user)
 
 
    print('users within lsoa...\t', time.time() - t1)

    return lsoa_users




#####################################################################################



def get_user_lsoas_paralel(args):
    

    user_coord_chunks = args[0]
    cityshape         = args[1]
    thread_id         = args[2]
    city              = args[3]
    outfolder         = args[4]
    nnn               = len(user_coord_chunks)


    fout = open(outfolder + '/venue_lsoa_attributes_' + str(thread_id), 'w')

    for ind, (user, coord) in enumerate(user_coord_chunks.items()):

        #if ind == 50: break
        if ind % 100 == 0: 
            print (thread_id, '\t', ind, '/', nnn)

        lat = float(coord[1])
        lng = float(coord[0])


        pnt      = Point(lng, lat)        
        query_df = cityshape[cityshape.contains(pnt)]

        if query_df.shape[0] == 1:
            lsoa = query_df.iloc[0]['lsoa11cd']
            fout.write (user + '\t' + lsoa + '\n')

    fout.close()



def get_users_lsoa(city, outroot, cityshape):
    
  
    print ('Get users lsoa...')

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



    outfolder = outroot + '/user_info/lsoa_user_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)




    Pros = [] 
    for i in range(0,num_threads):  
        user_coord_chunks = {k : users_homes[k] for k in users_chunks[i] }
        p = Process(target = get_user_lsoas_paralel, args=([user_coord_chunks, cityshape, i, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    files = os.listdir(outfolder)
    fout  = open(outroot  + '/user_info/user_lsoas.dat', 'w')
    fout.write('user\tlsoa\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            fout.write(line)
    fout.close()
            






def friendships_within_lsoa(lsoa_users, friends_list):
                

    t1 = time.time()
    print ('Get friendships within lsoa')

    users_lsoa = {}
    for lsoa, users in lsoa_users.items():
        for user in users:
            users_lsoa[user] = lsoa

        
    lsoa_friendships = {}    
        
        
    for user, friends in friends_list.items():

        
        if user in users_lsoa:

            user_lsoa = users_lsoa[user]

            if str(user_lsoa) != 0:

                for friend in friends:
                    if user_lsoa == users_lsoa[friend]:

                        friendship = '_'.join([user, friend])

                        if lsoa not in lsoa_friendships:
                            lsoa_friendships[user_lsoa] = set([friendship])
                        else:
                            lsoa_friendships[user_lsoa].add(friendship)

    

                    
        
    for lsoa in lsoa_users:
        if lsoa not in lsoa_friendships:
            lsoa_friendships[lsoa] = 0
        else:
            lsoa_friendships[lsoa] = len(lsoa_friendships[lsoa])
    
    print ('LSOA USERS: ', len(lsoa_users), '\t', time.time() - t1)
        
    return lsoa_friendships
        

    
    
''' =========================================================== '''
'''      get the network properties of the lsoa level nws       '''
''' =========================================================== '''

def get_lsoa_mininw_edges(lsoa_venues, edges_weights):
    
    t1 = time.time()
    print ('Create the LSOA level mini-network edge lists...')

    lsoa_edges = {}

    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):
        #if ind == 10: break

        for v1 in venues:
            for v2 in venues:
                if v1 != v2:
                    edge = '_'.join(sorted([v1,v2]))

                    if edge in edges_weights:

                        weight = edges_weights[edge]    

                        if lsoa not in lsoa_edges:
                            lsoa_edges[lsoa] = [(edge, weight)]
                        else:
                            lsoa_edges[lsoa].append((edge, weight))

    print ('LSOA level networks..\t', time.time() - t1)

    return lsoa_edges




''' =========================================================== '''
'''            get the stuff on the lsoa network level          '''
''' =========================================================== '''

def get_lsoa_nw_features(lsoa_venues, lsoa_edges, edges_weights):

    t1 = time.time()
    print ('Deriving the LSOA network features...')

    all_edges = [set(e.split('_')) for e in edges_weights.keys()]
    all_nodes = list(([ e.split('_')[0]  for e in edges_weights.keys()] + [ e.split('_')[1]  for e in edges_weights.keys()]))
    NNN       = len(all_nodes)

    edge_avg_weight_glb = np.mean(list(edges_weights.values()))                # global avg weight
    edge_density_glb    = len(all_edges) / ( NNN * (NNN - 1) / 2.0 )     # number of existing edges out 

    lsoa_weights_density = {}
    
    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):

        #if ind == 100: break

        # edge types
        #   - within the lsoa
        #   - on the boundary (one node wihtin, other outside)
        #   - outside of the lsoa (both nodes outside)
        # that didnt work out (computationally)  --> comparing lsoa level stuff to the global stuff

        edge_num               = 0.0            # number of edges within the lsoa
        node_num               = len(venues)    # number of nodes (venues) within the lsoa
        edge_avg_weight_in     = 0.0            # avg edge weight within the lsoa
        edge_density_in        = 0.0            # number of existing edges within lsoa out of possible ones


        venue_num = len(venues)

        if venue_num > 1 and lsoa in lsoa_edges:

            weights  = [e[1] for e in lsoa_edges[lsoa]]
            edges    = [e[0] for e in lsoa_edges[lsoa]]   
            nodes    = venues

            edge_num            = len(weights)    
            edge_avg_weight_in  = sum(weights) / edge_num 
            edge_density_in     = edge_num / ( venue_num * (venue_num - 1 ) / 2.0  )

            lsoa_weights_density[lsoa] = (edge_avg_weight_in, edge_density_in)
            

    print ('lsoa level network features\t', time.time() - t1)

    return lsoa_weights_density, edge_density_glb, edge_avg_weight_glb



def get_venues_features(lsoa_polygons,lsoa_local_friendships, lsoa_users, lsoa_venues, lsoa_num_users, lsoa_friendships, lsoa_weights_density, edge_density_glb, edge_avg_weight_glb, outfolder, city):

    venues_features = {}

    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):
 
        polygon     = lsoa_polygons[lsoa]
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
        lsoafriends = 0
        
        if lsoa in lsoa_weights_density:
            lsoa_weight = lsoa_weights_density[lsoa][0]
            lsoa_dens   = lsoa_weights_density[lsoa][1]
        else:
            lsoa_weight = 0.0
            lsoa_dens   = 0.0
            
        if lsoa in lsoa_users:
            livingthere = len(lsoa_users[lsoa])
            
            
        if lsoa in lsoa_num_users:
            usersnum = lsoa_num_users[lsoa]
            
        if lsoa in lsoa_friendships:
            friendships = len(lsoa_friendships)
            
        if lsoa in lsoa_local_friendships:
            lsoafriends = lsoa_local_friendships[lsoa]
  
        if area < 0.01:

            for venue in venues:

                feats = { 'bnds_lng0'       : bounds[0],
                          'bnds_lat0'       : bounds[1],
                          'bnds_lng1'       : bounds[2],
                          'bnds_lat1'       : bounds[3],
                          'bnds_length'     : polygon.length,
                          'bnds_area'       : polygon.area,
                          'lsoa_weight'     : lsoa_weight,
                          'lsoa_dens'       : lsoa_dens,
                          'lsoa_rel_weight' : lsoa_weight / edge_avg_weight_glb,
                          'lsoa_rel_dens'   : lsoa_dens   / edge_density_glb,
                          'userslikingnum'  : usersnum,
                          'friendships'     : friendships,
                          'livingthere'     : livingthere,
                          'lsoafriends'     : lsoafriends
                        }

                venues_features[venue] = feats

            

    
    filename = outfolder + 'networks/' + city  + '_LSOA_networkmeasures.csv'  
    df = pd.DataFrame.from_dict(venues_features, orient = 'index')
    df.to_csv(filename, sep = '\t')
    
    return df
    


    


def get_venue_lsoa_attributes(city, outfolder, bbox):
    

    cityshape                   = load_shp(city)
    #venues_coordinates          = get_venues_coordinates(city, outfolder)
    #lsoa_venues, lsoa_polygons  = get_lsoa_venues(cityshape, venues_coordinates, bbox, city, outfolder)


    lsoa_users             = get_users_lsoa(city, outfolder, cityshape)    


def get_lsoa_level_networks( city, outfolder, bbox ):

 #   city               = 'bristol'
 #   outfolder          = '../ProcessedData/' + city + '/'

    print ('================================\n==-- Start LSOA level stuff --==\n')

    get_venue_lsoa_attributes(city, outfolder, bbox)






 

    '''venues_users           = get_venues_users(outfolder, city)
    all_venues             = set([venue for venues in lsoa_venues.values() for venue in venues])
    edges_weights          = get_edge_weights2(city, outfolder, venues_users, lsoa_venues)    # node1_node2 -> weight
    nodes_edge_weights     = get_node_edge_list(edges_weights)     # node0 -> [(node1, w1), (node2, w2), ...]


    lsoa_edges   = get_lsoa_mininw_edges(lsoa_venues, edges_weights  )    # edges within the mini lsoa level networks
    venues_users = get_venues_users(outfolder, city)   
    friends_list = get_users_friends(outfolder, city)
   

    lsoa_local_friendships = friendships_within_lsoa(lsoa_users, friends_list)


    


    lsoa_weights_density, edge_density_glb, edge_avg_weight_glb  =  get_lsoa_nw_features(lsoa_venues, lsoa_edges, edges_weights)
    lsoa_num_users, lsoa_friendships                             = get_friendship_ties_within_lsoa(lsoa_venues, venues_users, friends_list)


    get_venues_features(lsoa_polygons, lsoa_local_friendships, lsoa_users, lsoa_venues, lsoa_num_users, lsoa_friendships, lsoa_weights_density, edge_density_glb, edge_avg_weight_glb, outfolder, city)      
    '''
   



