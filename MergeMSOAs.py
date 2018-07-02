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
    msoa_shp_df = gpd.read_file('full2/england_msoa_2011_gen.shp')

    if 'london' == city:
        

    return msoa_shp_df[msoa_shp_df['name'].str.contains(city.title())].to_crs({'init': 'epsg:4326'})  

'''
def load_shp(city):
    
    t1 = time.time()

    print ('Loading the shapefile...')
    msoa_shp_df = gpd.read_file('statistical-gis-boundaries-london/ESRI/MSOA_2004_London_High_Resolution.shp').to_crs({'init': 'epsg:4326'})  

    #return msoa_shp_df[msoa_shp_df['name'].str.contains(city.title())].to_crs({'init': 'epsg:4326'})  
    print ('Shapefile loaded\t', time.time() - t1)
    return msoa_shp_df





''' ========================================================================== '''
''' get the msoa id and the polygon of each coordinate (if its within the city '''
''' ========================================================================== '''



def coordinates_to_msoa(lats, lons, cityshape):
    
    poly = (0,0)
    
    try:
        pnt = Point(lons, lats)
        query_df = cityshape[cityshape.contains(pnt)]
        if query_df.shape[0] == 1:
            poly = (query_df.iloc[0]['msoa11cd'], query_df.iloc[0]['geometry'])
    except Exception as exception:
        pass
    
    return poly


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
'''              get the venues within msoa-s                   '''
''' =========================================================== '''

def get_msoa_venues_old(cityshape, venues_coordinates, bbox, city):

    t1 = time.time()
    print ('Converting (lat, long) to msoa-s...')
    
    msoa_venues  = {}
    msoa_polygons = {}

    nnn = len(venues_coordinates)



    for ind, (v, c) in enumerate(venues_coordinates.items()):


        lat = float(c[1])
        lng = float(c[0])
    

        if check_box(bbox, city, lat, lng):

            msoa, polygon = coordinates_to_msoa( lat, lng, cityshape )

            if msoa != 0:          
                       
                if msoa not in msoa_polygons:
                    msoa_polygons[msoa] = polygon    
                
                if msoa not in msoa_venues:
                    msoa_venues[msoa] = [v]
                else:
                    msoa_venues[msoa].append(v)

    print ('Coordinates converted to msoa-s\t', time.time() - t1)

    return msoa_venues, msoa_polygons







########################################################################


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        



def get_msoas_paralel(args):
    

    venues_coord_chunks = args[0]
    cityshape           = args[1]
    thread_id           = args[2]
    bbox                = args[3]
    city                = args[4]
    outfolder           = args[5]
    nnn                 = len(venues_coord_chunks)


    fout = open(outfolder + '/venue_msoa_attributes_' + str(thread_id), 'w')

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
                    msoa, polygon = (query_df.iloc[0]['MSOA_CODE'], query_df.iloc[0]['geometry'])

                   # print (lat, lng, msoa)

                    bounds  = polygon.bounds
                    lng0    = str(bounds[0])
                    lat0    = str(bounds[1])
                    lng1    = str(bounds[2])
                    lat1    = str(bounds[3])
                    length  = str(polygon.length)
                    area    = str(polygon.area)

                    fout.write ('\t'.join([venue, str(lng), str(lat), msoa, lng0, lat0, lng1, lat1, length, area]) + '\n')

                except:
                    pass

    fout.close()





def get_msoa_venues(cityshape, venues_coordinates, bbox, city, outfolder_):

    t1 = time.time()
    print ('Converting (lat, long) to msoa-s...')
    
    msoa_venues   = {}
    msoa_polygons = {}



    num_threads  = 40
    venues       = list(venues_coordinates.keys())
    venue_chunks = chunkIt(venues, num_threads)



    outfolder = outfolder_ + '/venues_info/msoa_venues_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)



    Pros = [] 
    for i in range(0,num_threads):  
        venues_coord_chunks = {k : venues_coordinates[k] for k in venue_chunks[i] }
        p = Process(target = get_msoas_paralel, args=([venues_coord_chunks, cityshape, i, bbox, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    files = os.listdir(outfolder)
    fout  = open(outfolder_  + '/venues_info/venues_msoa_full.dat', 'w')
    fout.write('venue\tlng\tlat\tmsoa\tlng0\tlat0\tlng1\tlat1\tlength\tarea\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            fout.write(line)
    fout.close()
            

    return msoa_venues, msoa_polygons



''' =========================================================== '''
'''         parse the edges of the venue sim nw                 '''
''' =========================================================== '''





def get_edge_weights2(city, outfolder, venues_users, msoa_venues):
    

    t1 = time.time()
    edges_weights2 = {}
    print ('Parsing venue similarity network edge list...')
    
    nnn = len(msoa_venues)    

    for ind, (msoa, venues) in enumerate(msoa_venues.items()):
                          
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
'''        get the lists of nghbnodes fir each node             '''
''' =========================================================== '''


    


''' =========================================================== '''
'''     get users stuff liking venues within the same msoa      '''
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
        



def get_friendship_ties_within_msoa(msoa_venues, venues_users, friends_list):
    
    t1 = time.time()
    print ('Get friendship ties within msoa')

    msoa_friendships = {}
    msoa_num_users   = {}
    
    nnn = len(msoa_venues)

    for ind, (msoa, venues) in enumerate(msoa_venues.items()):
        
#        if ind == 100 : break
 #       print (ind, '/', nnn)

        users = []
        for venue in venues:
            users += venues_users[venue]
    
    
        msoa_num_users[msoa] = len(users)
    
        for u1 in users:
            if u1 in friends_list:
                for u2 in users:
                    if u1 != u2:
                        if u2 in friends_list:
                            edge = '_'.join(sorted([u1, u2]))
    
                            if msoa not in msoa_friendships:
                                msoa_friendships[msoa] = [edge]
                            else:
                                msoa_friendships[msoa].append(edge)
    
    print('msoa friendship ties processed\t', time.time() - t1)    

    return msoa_num_users, msoa_friendships    




''' =========================================================== '''
'''        get users sutff living in the same msoa              '''
''' =========================================================== '''


def get_users_msoa_old(city, outroot, cityshape):
    
    
    t1 = time.time()
    print ('Get users msoa...')

    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'

    msoa_users = {}



    for ind, line in enumerate(open(infile)):
        user, lng, lat = line.strip().split('\t')


#        print (ind)
#        if ind == 100: break
        
        msoa, polygon = coordinates_to_msoa( float(lat), float(lng), cityshape ) 
        
        if msoa not in msoa_users:
            msoa_users[msoa] = [user]
        else:
            msoa_users[msoa].append(user)
 
 
    print('users within msoa...\t', time.time() - t1)

    return msoa_users




#####################################################################################



def get_user_msoas_paralel(args):
    

    user_coord_chunks = args[0]
    cityshape         = args[1]
    thread_id         = args[2]
    city              = args[3]
    outfolder         = args[4]
    nnn               = len(user_coord_chunks)


    fout = open(outfolder + '/venue_msoa_attributes_' + str(thread_id), 'w')

    for ind, (user, coord) in enumerate(user_coord_chunks.items()):

        #if ind == 50: break
        if ind % 100 == 0: 
            print (thread_id, '\t', ind, '/', nnn)

        lat = float(coord[1])
        lng = float(coord[0])


        pnt      = Point(lng, lat)        
        query_df = cityshape[cityshape.contains(pnt)]

        if query_df.shape[0] == 1:
            msoa = query_df.iloc[0]['msoa11cd']
            fout.write (user + '\t' + msoa + '\n')

    fout.close()



def get_users_msoa(city, outroot, cityshape):
    
  
    print ('Get users msoa...')

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



    outfolder = outroot + '/user_info/msoa_user_temp/' # + city + '_venues_users.dat'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)




    Pros = [] 
    for i in range(0,num_threads):  
        user_coord_chunks = {k : users_homes[k] for k in users_chunks[i] }
        p = Process(target = get_user_msoas_paralel, args=([user_coord_chunks, cityshape, i, city, outfolder], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



    files = os.listdir(outfolder)
    fout  = open(outroot  + '/user_info/user_msoas.dat', 'w')
    fout.write('user\tmsoa\n')
    for fn in files:
        for line in open(outfolder + '/' + fn):
            fout.write(line)
    fout.close()
            






def friendships_within_msoa(msoa_users, friends_list):
                

    t1 = time.time()
    print ('Get friendships within msoa')

    users_msoa = {}
    for msoa, users in msoa_users.items():
        for user in users:
            users_msoa[user] = msoa

        
    msoa_friendships = {}    
        
        
    for user, friends in friends_list.items():

        
        if user in users_msoa:

            user_msoa = users_msoa[user]

            if str(user_msoa) != 0:

                for friend in friends:
                    if user_msoa == users_msoa[friend]:

                        friendship = '_'.join([user, friend])

                        if msoa not in msoa_friendships:
                            msoa_friendships[user_msoa] = set([friendship])
                        else:
                            msoa_friendships[user_msoa].add(friendship)

    

                    
        
    for msoa in msoa_users:
        if msoa not in msoa_friendships:
            msoa_friendships[msoa] = 0
        else:
            msoa_friendships[msoa] = len(msoa_friendships[msoa])
    
    print ('msoa USERS: ', len(msoa_users), '\t', time.time() - t1)
        
    return msoa_friendships
        

    
    
''' =========================================================== '''
'''      get the network properties of the msoa level nws       '''
''' =========================================================== '''

def get_msoa_mininw_edges(msoa_venues, edges_weights):
    
    t1 = time.time()
    print ('Create the msoa level mini-network edge lists...')

    msoa_edges = {}

    for ind, (msoa, venues) in enumerate(msoa_venues.items()):
        #if ind == 10: break

        for v1 in venues:
            for v2 in venues:
                if v1 != v2:
                    edge = '_'.join(sorted([v1,v2]))

                    if edge in edges_weights:

                        weight = edges_weights[edge]    

                        if msoa not in msoa_edges:
                            msoa_edges[msoa] = [(edge, weight)]
                        else:
                            msoa_edges[msoa].append((edge, weight))

    print ('msoa level networks..\t', time.time() - t1)

    return msoa_edges




''' =========================================================== '''
'''            get the stuff on the msoa network level          '''
''' =========================================================== '''

def get_msoa_nw_features(msoa_venues, msoa_edges, edges_weights):

    t1 = time.time()
    print ('Deriving the msoa network features...')

    all_edges = [set(e.split('_')) for e in edges_weights.keys()]
    all_nodes = list(([ e.split('_')[0]  for e in edges_weights.keys()] + [ e.split('_')[1]  for e in edges_weights.keys()]))
    NNN       = len(all_nodes)

    edge_avg_weight_glb = np.mean(list(edges_weights.values()))                # global avg weight
    edge_density_glb    = len(all_edges) / ( NNN * (NNN - 1) / 2.0 )     # number of existing edges out 

    msoa_weights_density = {}
    
    for ind, (msoa, venues) in enumerate(msoa_venues.items()):

        #if ind == 100: break

        # edge types
        #   - within the msoa
        #   - on the boundary (one node wihtin, other outside)
        #   - outside of the msoa (both nodes outside)
        # that didnt work out (computationally)  --> comparing msoa level stuff to the global stuff

        edge_num               = 0.0            # number of edges within the msoa
        node_num               = len(venues)    # number of nodes (venues) within the msoa
        edge_avg_weight_in     = 0.0            # avg edge weight within the msoa
        edge_density_in        = 0.0            # number of existing edges within msoa out of possible ones


        venue_num = len(venues)

        if venue_num > 1 and msoa in msoa_edges:

            weights  = [e[1] for e in msoa_edges[msoa]]
            edges    = [e[0] for e in msoa_edges[msoa]]   
            nodes    = venues

            edge_num            = len(weights)    
            edge_avg_weight_in  = sum(weights) / edge_num 
            edge_density_in     = edge_num / ( venue_num * (venue_num - 1 ) / 2.0  )

            msoa_weights_density[msoa] = (edge_avg_weight_in, edge_density_in)
            

    print ('msoa level network features\t', time.time() - t1)

    return msoa_weights_density, edge_density_glb, edge_avg_weight_glb



def get_venues_features(msoa_polygons,msoa_local_friendships, msoa_users, msoa_venues, msoa_num_users, msoa_friendships, msoa_weights_density, edge_density_glb, edge_avg_weight_glb, outfolder, city):

    venues_features = {}

    for ind, (msoa, venues) in enumerate(msoa_venues.items()):
 
        polygon     = msoa_polygons[msoa]
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
        msoafriends = 0
        
        if msoa in msoa_weights_density:
            msoa_weight = msoa_weights_density[msoa][0]
            msoa_dens   = msoa_weights_density[msoa][1]
        else:
            msoa_weight = 0.0
            msoa_dens   = 0.0
            
        if msoa in msoa_users:
            livingthere = len(msoa_users[msoa])
            
            
        if msoa in msoa_num_users:
            usersnum = msoa_num_users[msoa]
            
        if msoa in msoa_friendships:
            friendships = len(msoa_friendships)
            
        if msoa in msoa_local_friendships:
            msoafriends = msoa_local_friendships[msoa]
  
        if area < 0.01:

            for venue in venues:

                feats = { 'bnds_lng0'       : bounds[0],
                          'bnds_lat0'       : bounds[1],
                          'bnds_lng1'       : bounds[2],
                          'bnds_lat1'       : bounds[3],
                          'bnds_length'     : polygon.length,
                          'bnds_area'       : polygon.area,
                          'msoa_weight'     : msoa_weight,
                          'msoa_dens'       : msoa_dens,
                          'msoa_rel_weight' : msoa_weight / edge_avg_weight_glb,
                          'msoa_rel_dens'   : msoa_dens   / edge_density_glb,
                          'userslikingnum'  : usersnum,
                          'friendships'     : friendships,
                          'livingthere'     : livingthere,
                          'msoafriends'     : msoafriends
                        }

                venues_features[venue] = feats

            

    
    filename = outfolder + 'networks/' + city  + '_msoa_networkmeasures.csv'  
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




def read_venues_msoa(outfolder, city):

    msoa_venues = {}
    infile      = outfolder + '/venues_info/venues_msoa_full.dat'

    for line in open(infile):
        if 'msoa' not in line:
            fields = line.strip().split('\t')
            venue, msoa = fields[0],fields[3]

            if msoa not in msoa_venues:
                msoa_venues[msoa] = [venue]
            else:
                msoa_venues[msoa].append(venue)

    return msoa_venues



def read_users_msoa(outfolder, city):

    msoa_users = {}
    infile     = outfolder + '/user_info/user_msoas.dat'

    for line in open(infile):
        if 'msoa' not in line:
            user, msoa = line.strip().split('\t')
          
            if msoa not in msoa_users:
                msoa_users[msoa] = [user]
            else:
                msoa_users[msoa].append(user)

    return msoa_users


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
'''             get and save venues and users msoa              '''
''' =========================================================== '''


def msoa_preproc(city, outfolder, bbox):
    

    cityshape                   = load_shp(city)
    venues_coordinates          = get_venues_coordinates(city, outfolder)
    msoa_venues, msoa_polygons  = get_msoa_venues(cityshape, venues_coordinates, bbox, city, outfolder)
  #  msoa_users                  = get_users_msoa(city, outfolder, cityshape)    


''' =========================================================== '''
'''                get msoa level network stuff                 '''
''' =========================================================== '''



def get_msoa_level_networks( city, outfolder, bbox ):

 
  
#    msoa_venues   = read_venues_msoa(outfolder, city)
#    msoa_users    = read_users_msoa(outfolder, city)
#    venues_users  = get_venues_users(outfolder, city)
#    edges_weights = get_edge_weights2(city, outfolder, venues_users, msoa_venues)    # node1_node2 -> weight




    nodes_edge_weights     = get_node_edge_list(edges_weights)     # node0 -> [(node1, w1), (node2, w2), ...]


    msoa_edges   = get_msoa_mininw_edges(msoa_venues, edges_weights  )    # edges within the mini msoa level networks
    venues_users = get_venues_users(outfolder, city)   
    friends_list = get_users_friends(outfolder, city)
   

    msoa_local_friendships = friendships_within_msoa(msoa_users, friends_list)





