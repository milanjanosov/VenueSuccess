import pandas as pd
import geopandas as gpd
from geopandas.geoseries import Point
import numpy as np





''' =========================================================== '''
'''             load the shapefile of UK                        '''
''' =========================================================== '''

def load_shp():
    
    print 'Loading the shapefile...'
    lsoa_shp_df = gpd.read_file('full2/england_lsoa_2011_gen.shp')
    return lsoa_shp_df[lsoa_shp_df['name'].str.contains(city.title())].to_crs({'init': 'epsg:4326'})  





''' ========================================================================== '''
''' get the LSOA id and the polygon of each coordinate (if its within the city '''
''' ========================================================================== '''

def coordinates_to_lsoa(lats, lons, cityshape):
    
    poly = (0,0)
    
    try:
        pnt = Point(lons, lats)
        query_df = cityshape[cityshape.contains(pnt)]
        if query_df.shape[0] == 1:
            poly = (query_df.iloc[0]['name'], query_df.iloc[0]['geometry'])
    except Exception as exception:
        pass
    
    return poly





''' =========================================================== '''
'''          parse the coordinates of the 4sqr venues           '''
''' =========================================================== '''

def get_venues_coordinates(city, outfolder):

    print 'Parsing venue coordinates...'

    venues_coordinates = {}

    for ind, line in enumerate(open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat')):
        #if ind == 10: break
        fields = line.strip().split('\t')
        venues = [fff.split(',') for fff in fields[1:]]

        for v in venues:
            venues_coordinates[v[0]] = (float(v[1]), float(v[2]))

    return venues_coordinates





''' =========================================================== '''
'''              get the venues within lsoa-s                   '''
''' =========================================================== '''

def get_lsoa_venues(venues_coordinates):

    print 'Converting (lat, long) to LSOA-s...'
    
    lsoa_venues  = {}
    lsoa_polygons = {}

    for ind, (v, c) in enumerate(venues_coordinates.items()):
        #if ind == 50: break
        
        lsoa, polygon = coordinates_to_lsoa( c[1], c[0], cityshape )

        if lsoa != 0:          
                   
            if lsoa not in lsoa_polygons:
                lsoa_polygons[lsoa] = polygon    
            
            if lsoa not in lsoa_venues:
                lsoa_venues[lsoa] = [v]
            else:
                lsoa_venues[lsoa].append(v)
        
    return lsoa_venues, lsoa_polygons





''' =========================================================== '''
'''         parse the edges of the venue sim nw                 '''
''' =========================================================== '''

def get_edge_weights(city, outfolder):

    print 'Parsing venue similarity network edge list...'
    
    edges_weights = {}
    all_edges   = set()
    
    for ind,line in enumerate(open(outfolder + 'networks/gephi/' + city + '_venues_similarity_edges.dat')):
       # if ind == 10: break
        if 'Target' not in line:
            source, target, distance, weight, t = line.strip().split('\t')
            
            weight = float(weight)
            edge   = '_'.join(sorted([source, target]))
            edges_weights[edge] = weight
            all_edges.add(source)
            all_edges.add(target)
            
    return edges_weights





''' =========================================================== '''
'''        get the lists of nghbnodes fir each node             '''
''' =========================================================== '''

def get_node_edge_list(edges_weights):

    print 'Listing each nodes neighbours and those edge weights...'

    # for each node list the edges (weights) in which they are present
    nodes_edge_weights = {}
    
    for ind, (e, w) in enumerate(edges_weights.items()):
    
        #if ind == 100: break
            
        e1, e2 = e.split('_')
        
        if e1 not in nodes_edge_weights:
            nodes_edge_weights[e1] = [(e2, w)]
        else:
            nodes_edge_weights[e1].append((e2, w))

                    
        if e2 not in nodes_edge_weights:
            nodes_edge_weights[e2] = [(e1, w)]
        else:
            nodes_edge_weights[e2].append((e1, w))
        
    return nodes_edge_weights





''' =========================================================== '''
'''      get the network properties of the lsoa level nws       '''
''' =========================================================== '''

def get_lsoa_mininw_edges(lsoa_venues):
    
    print 'Create the LSOA level mini-network edge lists...'

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

    return lsoa_edges





''' =========================================================== '''
'''            get the stuff on the lsoa network level          '''
''' =========================================================== '''

def get_lsoa_nw_features(lsoa_edges, edges_weights):

    print 'Deriving the LSOA network features...'

    all_edges = [set(e.split('_')) for e in edges_weights.keys()]
    all_nodes = list(([ e.split('_')[0]  for e in edges_weights.keys()] + [ e.split('_')[1]  for e in edges_weights.keys()]))
    NNN       = len(all_nodes)

    edge_avg_weight_glb = np.mean(edges_weights.values())                # global avg weight
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
            

    return lsoa_weights_density, edge_density_glb, edge_avg_weight_glb



def get_venues_features(lsoa_weights_density, edge_density_glb, edge_avg_weight_glb):

    print 'Summing up the LSOA level network features...'

    venues_features = {}

    for ind, (lsoa, venues) in enumerate(lsoa_venues.items()):
 
        polygon = lsoa_polygons[lsoa]
        bounds  = polygon.bounds
        lng0    = bounds[0]
        lat0    = bounds[1]
        lng1    = bounds[2]
        lat1    = bounds[3]
        length  = polygon.length
        area    = polygon.area
        
        if lsoa in lsoa_weights_density:
            lsoa_weight = lsoa_weights_density[lsoa][0]
            lsoa_dens   = lsoa_weights_density[lsoa][1]
        else:
            lsoa_weight = 0.0
            lsoa_dens   = 0.0
                
        print lsoa_weight, lsoa_dens
            

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
                      'lsoa_rel_dens'   : lsoa_dens   / edge_density_glb
                    }

            venues_features[venue] = feats

    
    filename = outfolder + 'networks/' + city  + '_LSOA_networkmeasures.csv'  
    df = pd.DataFrame.from_dict(venues_features, orient = 'index')
    df.to_csv(filename, sep = '\t')
    




city               = 'bristol'
outfolder          = '../ProcessedData/' + city + '/'

cityshape                   = load_shp()
venues_coordinates          = get_venues_coordinates(city, outfolder)
lsoa_venues, lsoa_polygons  = get_lsoa_venues(venues_coordinates)
all_venues                  = set([venue for venues in lsoa_venues.values() for venue in venues])
edges_weights               = get_edge_weights(city, outfolder)     # node1_node2 -> weight
nodes_edge_weights          = get_node_edge_list(edges_weights)     # node0 -> [(node1, w1), (node2, w2), ...]
lsoa_edges                  = get_lsoa_mininw_edges(lsoa_venues)    # edges within the mini lsoa level networks


lsoa_weights_density, edge_density_glb, edge_avg_weight_glb  =  get_lsoa_nw_features(lsoa_edges, edges_weights)

get_venues_features(lsoa_weights_density, edge_density_glb, edge_avg_weight_glb)      










