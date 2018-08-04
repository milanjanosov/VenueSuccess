import numpy as np
from igraph import Graph
from collections import OrderedDict
import pandas as pd
import mpu
import os
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from ParseJsons import create_folder
from ParseJsons import check_box
import sys
from multiprocessing import Process, Manager
sys.path.append("./backboning")
import backboning
sys.path.append("..")
import ParseInput
from shapely.geometry import shape
from pyproj import Proj






''' =========================================================== '''
''' =============   CALC GEODISTANCE BETWEEN NODS ============= '''
''' =========================================================== '''




def add_distances_to_edges(G, avg_dist):

    distances     = []
    inv_distances = []
    exp_dist      = []
    grav_dist     = []


    for i, e in enumerate(G.es()):

    
        target = G.vs[e.target]
        source = G.vs[e.source]

        if target['name'] != source['name']:

            target_loc = target['location']           
            source_loc = source['location']      

       
        dist    =  mpu.haversine_distance((target_loc[1], target_loc[0]), (source_loc[1], source_loc[0]))
        if dist == 0: dist = 0.0000000000000000000001


        distances.append(     dist       )
        inv_distances.append( dist**(-1) )
        grav_dist.append(     dist**(-2) )
        exp_dist.append(      math.exp( - 1.0 * dist / avg_dist) )


    G.es['distances']      = distances
    G.es['inv_distances']  = inv_distances
    G.es['grav_distances'] = grav_dist
    G.es['exp_distances']  = exp_dist

 

def get_network_stats(G, city, outfolder, infile):



    f, ax = plt.subplots(1, 3, figsize=(20, 5))

    N    = G.degree_distribution().n
    mean = round(G.degree_distribution().mean, 2)
    sd   = round(G.degree_distribution().sd, 2)
    xs, ys = zip(*[(left, count) for left, _, count in  G.degree_distribution().bins()])

 
    ax[0].bar(xs, ys)
    ax[0].set_title('Degree distribution, N = ' + str(N) + ', mean = ' + str(mean) + ', std = ' + str(sd) )
    ax[0].set_xlabel('Node degree')
    ax[0].set_ylabel('Degree distribution')
    ax[0].set_yscale('log')


    distances = G.es['distances']
    ax[1].hist(distances, bins = 50)
    ax[1].set_yscale('log')
    ax[1].set_title('Distance distribution, mean = ' + str(round(np.mean(distances), 2)) + ', std = ' + str(round(np.std(distances),2)) )
    ax[1].set_xlabel('Distance between pairs of users within the city')
    ax[1].set_ylabel('Distance distribution')


    try:    
        weights = G.es['weight']
        ax[2].hist(G.es['weight'], bins = 50)
        ax[2].set_yscale('log')
        ax[2].set_title('Distance distribution, mean = ' + str(round(np.mean(weights), 2)) + ', std = ' + str(round(np.std(weights),2)) )
        ax[2].set_xlabel('Edge weights (Jaccard sim)')
        ax[2].set_ylabel('Weight distribution')

    except:
        pass    


    plt.suptitle(city, fontsize = 18)    
    plt.show()   
    plt.savefig(outfolder + 'figures/network_data/' + city + infile + '.png')

   
# plt.close()





''' =========================================================== '''
''' ======= USERS FULL FRIENDSHIP NW AND GEOTAGGED NW  ======== '''
''' =========================================================== '''
def get_gephi_new(G, outfolder, outname):

       
    f = open( outfolder + 'networks/gephi/' + outname + '_edges.dat', 'w')
    f.write('Source' + '\t' + 'Target' + '\t' + 'distances'  + '\t' + 'inv_distances' + '\t' + 'grav_distances' + '\t' + 'exp_distances' + '\t' + 'Weight_J' + '\t' + 'Weight_A' + '\t' + 'Type' + '\n')      
    for e in G.es():

        try:
            f.write(G.vs[e.target]['name']+'\t'+G.vs[e.source]['name']+'\t'+str(e['distances'])+'\t'+str(e['inv_distances'])+'\t'+str(e['grav_distances'])+'\t'+str(e['exp_distances'])+'\t'+str(e['weight'])+'\t'+str(e['weight_a'])+ '\tundirected' + '\n')
        except:
            f.write(G.vs[e.target]['name']+'\t'+G.vs[e.source]['name']+'\t'+str(e['distances'])+'\t'+str(e['inv_distances'])+'\t'+str(e['grav_distances'])+'\t'+str(e['exp_distances'])+'\t'+str(1)+'\t'+str(1)+'\tundirected' + '\n')
            pass


    f.close()

    
    g = open( outfolder + 'networks/gephi/' + outname + '_nodes.dat', 'w')
    g.write('ID' + '\t' + 'Label' +'\t'+  'lon' + '\t' + 'lat' + '\n')    
    for n in G.vs():
        g.write(n['name'] + '\t' + n['name'] + '\t' + str(n['location'][0]) + '\t' + str(n['location'][1]) + '\n')
    g.close()

 


def get_user_user_friendship_network_igraph(city, outfolder, infile):


    print 'Start creating the friendship network...'

    filename  = outfolder + '/user_info/' + city  + '_users_friends.lgl' 
    all_users = set([line.strip().split('\t')[0] for line in open(infile)])
    fout        = open(outfolder + '/user_info/' + city  + '_users_geo_friends.lgl', 'w')
    users_nghbs = {}


    # get the coordinates
    users_location = {}
    for line in open(infile):
        user, lng, lat       = line.strip().split('\t')
        if user in all_users:
            users_location[user] = (float(lng), float(lat))    



    with open (filename, "r") as myfile:
        data = str(myfile.read()).strip().split('#')


    for ind, d in enumerate(data):
     
        users = d.split('\n')
        user  = users[0].replace(' ', '')
    
        if user in all_users:

            nghbs = [uuu for uuu in users[1:] if uuu in all_users]
     
            if len(nghbs) > 0:
                users_nghbs[user] = nghbs
         
 
    for user, friends in users_nghbs.items():
        if len(friends) > 1:
            fout.write('#' + user + '\n')
            for friend in list(set(friends)):
                fout.write(friend + '\n')
    fout.close()


    G_geo   = Graph.Read_Lgl(outfolder + '/user_info/' + city  + '_users_geo_friends.lgl', names = True, weights = False, directed = False) 
    G_geo.vs['location'] = [users_location[g['name']] for g in G_geo.vs()]
    

    nnn    = len(users_location)

    movavg = 0
    ijk    = 0

    for user1 in G_geo.vs():
        coord1 = user1['location']
        for user2 in G_geo.vs():
            coord2   = user2['location']
            pairdist = mpu.haversine_distance((coord1[1], coord1[0]), (coord2[1], coord2[0]))
            movavg   = movavg + ( pairdist - movavg ) / float( (ijk + 1) )
            ijk     += 1


    
    folderout = outfolder + 'networks/gephi/distances/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    fffout = open(folderout + '/friendship_avg_distance.dat', 'w')
    fffout.write(str(movavg) + '\n')
    fffout.close()

    #fffout = open(folderout + '/users_friendship_distances.dat', 'w')
    #fffout.write( '\n'.join([str(d) for d in pairdists]) + '\n' )
    #fffout.close()




    # add and calc distances

    add_distances_to_edges(G_geo, movavg)

    print 'Friendship network done.'
    
    return G_geo




''' =========================================================== '''
''' =======        GET USERS SIMILARITY NETWORK        ======== '''
''' =========================================================== '''






def get_user_user_similarity_network_igraph(city, outfolder, infile):


    T1 = time.time()




    # parse the data
    filename     = outfolder + 'user_info/' + city  + '_users_venues.dat'  
    venues_users = {}

    for ind, line in enumerate(open(filename)):
        fields = line.strip().split('\t')
        user   = fields[0]
        venues = fields[1:]

        for venue in venues:

            if venue not in venues_users:
                venues_users[venue] = [user]
            else:
                venues_users[venue].append(user)
            









    print 'Start creating the users\'s similarity network...'

    # get the coordinates
    users_location = {}
    for line in open(infile):
        user, lng, lat = line.strip().split('\t')
        users_location[user] = (float(lng), float(lat))


    # parse the files
    users_venfn  = outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat'  
    users_venues = {}
    users        = []


    for line in open(users_venfn):
        fields = line.strip().split('\t')
        user   = fields[0]
        if user in users_location:
            venues = set(fields[1:])
            users_venues[user] = venues
            users.append(user)


    # buld the network
    edges       = []
    weights     = []
    weights_a   = []
    all_users   = set()
    movavg      = 0  


    nnn = len(users)
    ijk = 0


    for ind, user1 in enumerate(users):

        #if ind == 3: break

        print ind, '/', nnn
        all_users.add(user1)

        coord1 = users_location[user1]

        for user2 in users:

            if user1 != user2:

                all_users.add(user2)


                coord2 = users_location[user2]
                tau    = list(users_venues[user1].intersection(users_venues[user2]))
                w      = len(tau)

                pairdist = mpu.haversine_distance((coord1[1], coord1[0]), (coord2[1], coord2[0]))



                movavg   = movavg + ( pairdist - movavg ) / float( (ijk + 1) )
                ijk      += 1

                if len(tau) > 0 and 'user' not in user1 and 'user' not in user2:      

                    norm   = len(users_venues[user1].union(users_venues[user2]))
                    w      = float(w) / norm
                    w_a    = sum([ 1.0/(math.log(len(venues_users[u.split(',')[0]]))) for u in tau])

                    edges.append((user1, user2))
                    weights.append(w)
                    weights_a.append(w)





    folderout = outfolder + 'networks/gephi/distances/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    fffout = open(folderout  + '/users_sim_avg_distance.dat', 'w')
    fffout.write(str(movavg) + '\n')
    fffout.close()

    #fffout = open(folderout + '/users_sim_distances.dat', 'w')
    #fffout.write( '\n'.join([str(d) for d in pairdists]) + '\n' )
    #fffout.close()



    G = Graph()
    G.add_vertices(list([u for u in all_users]))
    G.add_edges(edges)
    locations         = [users_location[g['name']] for g in G.vs()] 
    G.es['weight']    = weights   
    G.es['weight_a']  = weights_a 
    G.vs['location']  = locations

 
    add_distances_to_edges(G, movavg)
 
    print 'Users\'s similarity network done.'
    print time.time() - T1    

    return G
   
    


''' =========================================================== '''
''' =======     GET THE VENUES SIMILARITY NETWORK      ======== '''
''' =========================================================== '''

def get_venue_venue_similarity_network_igraph(city, outfolder, infile, bbox):


    T1 = time.time()
    print 'Start creating the venues\'s similarity network...'

    # parse the data
    filename     = outfolder + 'user_info/' + city  + '_users_venues.dat'  
    venues_users = {}
    vv = []

    for ind, line in enumerate(open(filename)):
        fields = line.strip().split('\t')
        user   = fields[0]
        venues = fields[1:]

        for venue in venues:

            if venue not in venues_users:
                venues_users[venue] = [user]
            else:
                venues_users[venue].append(user)
            
    
        vv += venues

    


    ### get users venues
    users_venfn    = outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat'  
    users_venues   = {}
    users          = []
    users_location = {}

    for line in open(infile):
        user, lng, lat = line.strip().split('\t')
        users_location[user] = (float(lng), float(lat))

    for line in open(users_venfn):
        fields = line.strip().split('\t')
        user   = fields[0]
        if user in users_location:
            venues = set(fields[1:])
            users_venues[user] = venues
            users.append(user)







    ### get venues locations
    venues_location = {}
    for  line in open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat'):
        fields = line.strip().split('\t')
        user   = fields[0]
        venues = fields[1:]
    
        for venue in venues:
            venid, lng, lat, cat = venue.split(',')

            if venid in venues_users and check_box(bbox, city, float(lat), float(lng)):
                venues_location[venid] = (float(lng), float(lat))   
 

   
    # build the nw
    G = Graph()
    
    edges      = []
    weights    = []
    weights_a  = []
    all_venues = set()
    pairdists  = []
    movavg     = 0

    venueslist = venues_location.keys() 
    nnn = len(venueslist)


    for ind, venue1 in enumerate(venueslist):

        print ind, '/', nnn


        all_venues.add(venue1)

        coord1 = venues_location[venue1]


        for venue2 in venueslist:

            if venue1 != venue2:

                all_venues.add(venue2)
    
                coord2   = venues_location[venue2]
                tau      = set(venues_users[venue1]).intersection(set(venues_users[venue2]))
                w        = float(len(tau))
                pairdist = mpu.haversine_distance((coord1[1], coord1[0]), (coord2[1], coord2[0]))
                movavg   = movavg + ( pairdist - movavg ) / float( (ind + 1) )


                if len(tau) > 0:

                    norm    = len(  set(venues_users[venue1]).union(set(venues_users[venue2] )) )
                    w       = w/norm
                    w_a     = sum([ 1.0/(math.log(len(users_venues[u]))) for u in tau if u in users_venues])

                    edges.append((venue1, venue2))
                    weights.append(w)
                    weights_a.append(w_a)

      


    folderout = outfolder + 'networks/gephi/distances/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    fffout = open(folderout + '/venues_sim_avg_distance.dat', 'w')
    fffout.write(str(movavg) + '\n')
    fffout.close()


    all_venues = list(all_venues)
    locations  = [venues_location[venue] if venue in venues_location else 'nan' for venue in all_venues] 


    G.add_vertices(all_venues)
    G.add_edges(edges)
    G.es['weight']   = weights
    G.es['weight_a'] = weights_a
    G.vs['location'] = [venues_location[g['name']] for g in G.vs()]  
    add_distances_to_edges(G, movavg)
    
    print movavg
    print 'Venue\'s similarity network done.'
    print time.time() - T1    
    return G






''' =========================================================== '''
''' ==============   DO THE BACKBONE FILTERING    ============= '''
''' =========================================================== '''



def transform_gephi_to_backbone(outfolder, outname, nc_threshold):


    fnin   = outfolder + 'networks/gephi/' + outname + '_edges.dat'

    if not os.path.exists(outfolder + 'networks/gephi/backbone'):
        os.makedirs(outfolder + 'networks/gephi/backbone')

    fnout_inv  = outfolder + 'networks/gephi/backbone/backboneformat_' + outname + 'inv_distances.dat'
    fnout_grav = outfolder + 'networks/gephi/backbone/backboneformat_' + outname + 'grav_distances.dat'
    fnout_exp  = outfolder + 'networks/gephi/backbone/backboneformat_' + outname + 'exp_distances.dat'
    fnout_w    = outfolder + 'networks/gephi/backbone/backboneformat_' + outname + 'weight.dat'
    fnout_wa   = outfolder + 'networks/gephi/backbone/backboneformat_' + outname + 'weight_a.dat'


    fout_inv = open(fnout_inv, 'w')
    fout_inv.write('src\ttrg\tnij\n')

    fout_grav = open(fnout_grav, 'w')
    fout_grav.write('src\ttrg\tnij\n')

    fout_exp = open(fnout_exp, 'w')
    fout_exp.write('src\ttrg\tnij\n')

    fout_w = open(fnout_w, 'w')
    fout_w.write('src\ttrg\tnij\n')

    fout_wa = open(fnout_wa, 'w')
    fout_wa.write('src\ttrg\tnij\n')




    print 'Start reading the edge list...'

    for ind, line in enumerate(open( fnin )):
        if ind % 10000 == 0: 
            print ind
#        if ind == 10 : break
       
        if 'Source' not in line:
       
            src, trg, distances, inv_distances, grav_distances, exp_distances, weight, weight_a, typee = line.strip().split('\t')

            fout_inv.write(  src + '\t' + trg + '\t' + str(float(inv_distances))  + '\n')
            fout_grav.write( src + '\t' + trg + '\t' + str(float(grav_distances)) + '\n')
            fout_exp.write(  src + '\t' + trg + '\t' + str(float(exp_distances))  + '\n')
            fout_w.write(    src + '\t' + trg + '\t' + str(float(weight))         + '\n')
            fout_wa.write(   src + '\t' + trg + '\t' + str(float(weight_a))       + '\n')


    fout_inv.close()
    fout_grav.close()
    fout_exp.close()
    fout_w.close()
    fout_wa.close()


    real_table_inv  = pd.read_csv(fnout_inv,  sep = "\t")
    real_table_grav = pd.read_csv(fnout_grav, sep = "\t")
    real_table_exp  = pd.read_csv(fnout_exp,  sep = "\t")
    real_table_w    = pd.read_csv(fnout_w,    sep = "\t")
    real_table_wa   = pd.read_csv(fnout_wa,   sep = "\t")


    #table_df   = backboning.disparity_filter(real_table, undirected = True)
    table_nc_inv  = backboning.noise_corrected(real_table_inv,  undirected = True)
    table_nc_grav = backboning.noise_corrected(real_table_grav, undirected = True)
    table_nc_exp  = backboning.noise_corrected(real_table_exp,  undirected = True)
    table_nc_w    = backboning.noise_corrected(real_table_w,    undirected = True)
    table_nc_wa   = backboning.noise_corrected(real_table_wa,   undirected = True)


    '''
    ffout = open(outfolder + 'networks/gephi/COMPARE_DF_thresholds_' + outname + '.dat', 'w')
    for df_threshold in [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96]:
        print 'DF thresholding', df_threshold
        bb_vespignani = backboning.thresholding(table_df, df_threshold)
        df_edgenum = len(bb_vespignani['src'])
        df_nodenum = len(set( list(bb_vespignani['src']) + list(bb_vespignani['trg'])))
        ffout.write(str(df_threshold) + '\t' + str(df_edgenum) + '\t' + str(df_nodenum) + '\n')
    ffout.close()
    fout_df = open(outfolder + 'networks/gephi/DF_BACKBONE_' + str( df_threshold ) + '_' + outname + '_edges.dat', 'w')
    print "Writing DF Backbone"
    bb_vespignani.to_csv(fout_df, sep = '\t', index = False)  
    '''


    bb_neffke_inv  = backboning.thresholding(table_nc_inv,  nc_threshold)
    bb_neffke_grav = backboning.thresholding(table_nc_grav, nc_threshold)
    bb_neffke_exp  = backboning.thresholding(table_nc_exp,  nc_threshold)
    bb_neffke_w    = backboning.thresholding(table_nc_w,    nc_threshold)
    bb_neffke_wa   = backboning.thresholding(table_nc_wa,   nc_threshold)

    #nc_edgenum = len(bb_neffke['src'])
    # = len(set( list(bb_neffke['src']) + list(bb_neffke['trg'])))
    #dens       =  nc_edgenum / (nc_nodenum**2/2.0)
    #print "Writing the NC Backbone with threshold  ", nc_threshold, ', node number ', nc_nodenum, '  and density ', dens Q
  
    fout_nc_inv  = open(outfolder + 'networks/gephi/backbone/NC_BACKBONE_inv_'  + str( nc_threshold ) + '_' + outname + '_edges.dat', 'w')
    fout_nc_grav = open(outfolder + 'networks/gephi/backbone/NC_BACKBONE_grav_' + str( nc_threshold ) + '_' + outname + '_edges.dat', 'w')
    fout_nc_exp  = open(outfolder + 'networks/gephi/backbone/NC_BACKBONE_exp_'  + str( nc_threshold ) + '_' + outname + '_edges.dat', 'w')
    fout_nc_w    = open(outfolder + 'networks/gephi/backbone/NC_BACKBONE_w_'    + str( nc_threshold ) + '_' + outname + '_edges.dat', 'w')
    fout_nc_wa   = open(outfolder + 'networks/gephi/backbone/NC_BACKBONE_wa_'   + str( nc_threshold ) + '_' + outname + '_edges.dat', 'w')

    bb_neffke_inv.to_csv(fout_nc_inv,   sep = '\t', index = False)
    bb_neffke_grav.to_csv(fout_nc_grav, sep = '\t', index = False)
    bb_neffke_exp.to_csv(fout_nc_exp,   sep = '\t', index = False)
    bb_neffke_w.to_csv(fout_nc_w,       sep = '\t', index = False)
    bb_neffke_w.to_csv(fout_nc_wa,      sep = '\t', index = False)



def create_igraphnw_from_backbone(outfolder, inname, tipus, infile, dist_type, thresh = ''):


    print 'Creating backbone igraph network ' + tipus

    ininfile = outfolder + 'networks/gephi/backbone/' + tipus + '_BACKBONE_'+ dist_type + '_' + thresh + '_' + inname + '_edges.dat'
    outfile  = outfolder + 'networks/gephi/backbone/' + tipus + '_IGRAPH_'  + dist_type + '_' + thresh + '_' + inname + '_edges.dat'


    print outfile

    # get the edges
    fout = open(outfile, 'w')
    for line in open(ininfile):
        fout.write( '\t'.join(line.strip().split('\t')[0:3]) + '\n') 
    fout.close()


    G = Graph.Read_Ncol(outfile, weights = True, directed=False)


    # get the nodes

    users_location = {}
    for line in open(infile):
        user, lng, lat       = line.strip().split('\t')
        users_location[user] = (float(lng), float(lat))    

 
    for ind, v in enumerate(G.vs()):
        if v['name']  == 'src': G.delete_vertices(v.index)
        if v['name']  == 'trg': G.delete_vertices(v.index)
        try:
            v['location'] = users_location[v['name']]
        except:
            G.delete_vertices(v.index)



    avgdist = [float(line.strip()) for line in open(outfolder + 'networks/gephi/distances/' + inname.split('_',1)[1] + '_avg_distance.dat')][0]


  

    add_distances_to_edges(G, avgdist)

    
    print 'Igraph transf\t', tipus, 'threshold: ', thresh, '  ', '  nodes:', len(G.vs()), '   edges:', len(G.es())

    return G







def create_igraphnw_from_backbone_for_venues(outfolder, inname, tipus, dist_type, infile, thresh = ''):


    print 'Creating backbone igraph network ' + tipus



    ininfile = outfolder + 'networks/gephi/backbone/' + tipus + '_BACKBONE_' + dist_type + '_' + thresh + '_' + inname + '_edges.dat'
    outfile  = outfolder + 'networks/gephi/backbone/' + tipus + '_IGRAPH_'   + dist_type + '_' + thresh + '_' + inname + '_edges.dat'


    # get the edges
    fout = open(outfile, 'w')
    for line in open(ininfile):
        fout.write( '\t'.join(line.strip().split('\t')[0:3]) + '\n') 
    fout.close()


    G = Graph.Read_Ncol(outfile, weights = True, directed=False)


    # get the nodes
    venues_location = {}
    for  line in open(outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat'):
        fields = line.strip().split('\t')
        user   = fields[0]
        venues = fields[1:]
        for venue in venues:    
            venid, lng, lat, cat = venue.split(',')
            venues_location[venid] = (float(lng), float(lat))   



    for ind, v in enumerate(G.vs()):
        if v['name']  == 'src': G.delete_vertices(v.index)
        if v['name']  == 'trg': G.delete_vertices(v.index)

        try:
            v['location'] = venues_location[v['name']]
        except:
            G.delete_vertices(v.index)

    print 'Igraph transf\t', tipus, 'threshold: ', thresh, '  ', '  nodes:', len(G.vs()), '   edges:', len(G.es())

    avgdist = [float(line.strip()) for line in open(outfolder + 'networks/gephi/distances/' + inname.split('_',1)[1] + '_avg_distance.dat')][0]
    add_distances_to_edges(G, avgdist)


    return G












''' =========================================================== '''
''' =========== DEFINE DISTANCE BASED NW MEASURES  ============ '''
''' =========================================================== '''


### 1/nneighbours \sum neighbors_distances
def social_stretch(G, node, neighbors, dist_type):

    return sum([G.es[G.get_eid(node.index, neighbor)][dist_type] for neighbor in neighbors if node.index != neighbor]) / (float(len(neighbors)))
  

### see if 3 nodes are triangle, if yes, then avg length within triangle
#def triangle_size(G, node, neighbors, dist_type):
#
#    triangle_edges = []
#
#    for n1 in neighbors:
#        for n2 in neighbors:
#
#            if n1 != n2:
#
#                try:
#                    triangle_edges.append(G.es[G.get_eid(n1, n2)][dist_type])
#                except:     
#                    pass
#
#    if len(triangle_edges) > 0:
#        return np.mean(triangle_edges)
#    else:
#        return 99999999999.9


### the size of the polygon of the users friends-neighbors
### https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python square meters 
def geo_size_of_ego(G, node, neighbors):


    lat, lon = zip(*[G.vs[neighbor]['location'] for neighbor in neighbors if 'nan' not in str(G.vs[neighbor]['location'])])

    if len(lat) > 2:

            pa   = Proj("+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55")
            x, y = pa(lon, lat)
            cop  = {"type": "Polygon", "coordinates": [zip(x, y)]}
  
            return shape(cop).area / 1000000.0

    else:

        return  0.0



def geo_stdev_of_ego(G, node, neighbors, dist_type):

    return np.std([G.es[G.get_eid(node.index, neighbor)][dist_type] for neighbor in neighbors if node.index != neighbor]) / (float(len(neighbors)))
  

def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


#def get_venue_homogenity(G, city, outfolder):
#    
#    users_venues = {} 
#
#    for line in open( outfolder + '/user_info/' + city + '_user_venues_full_locals_filtered.dat'):
#        fields                  = (line.strip().split('\t'))
#        users_venues[fields[0]] = set([ vv.split(',')[0] for vv in fields[1:]])
#
#    
#    venues_homogenity = {}
#    for line in open(outfolder + '/venues_info/' + city + '_venues_users.dat'):
#        fields = (line.strip().split('\t'))
#        venue  = fields[0]
#        users  = fields[1:] 
#
#        for u1 in users:
#            for u2 in users:
#                sim_u1_u2 = jaccard(users_venues[u1], users_venues[u2])
#
#        venues_homogenity[venue] = np.mean([jaccard(users_venues[u1], users_venues[u2]) for u1 in users for u2 in users if u1 in users_venues and u2 in users_venues])
#
#    return [venues_homogenity[g['name']] for g in G.vs()]



''' =========================================================== '''
''' ============ CALC THE NETWORK CENTRALITIE MEAS  =========== '''
''' =========================================================== '''


def get_distance_measures_network(G, dist_type, tipus):


    if dist_type is None:
        dist_type_ = ''
    else:
        dist_type_ = dist_type

    t1 = time.time()
    betweennesses  = G.betweenness(                   weights = dist_type)
    print '\n\n' +  tipus  + '  betweennes - ' +  dist_type_, '\t', round(time.time() - t1, 2), ' seconds'

    t1 = time.time()
    closenesses    = G.closeness(                     weights = dist_type)
    print tipus + '  closeness -  ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds'

    t1 = time.time()
    clusterings    = G.transitivity_local_undirected( weights = dist_type)
    print tipus + '  clustering - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds' 

    t1 = time.time()
    strengthes     = G.strength(                      weights = dist_type)
    print tipus + '  strentgh - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds' 

    t1 = time.time()
    pageranks      = G.pagerank(                      weights = dist_type)
    print tipus + '  pagerank - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds' 

    t1 = time.time()
    eigenvectors   = G.eigenvector_centrality(        weights = dist_type)
    print tipus + '  eigenvector - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds' 

    #if 'friend' in tipus: 
    t1 = time.time()
    constraint     = G.constraint(                    weights = dist_type)
    print tipus + '  constraint - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds'  , '\n'


    return betweennesses, closenesses, clusterings, strengthes, pageranks, eigenvectors, constraint





def populate_node_attributes(vertice_attributes, G, dist_type, degrees, neighborhood_sizes, betweennesses, closenesses, clusterings, strengthes, pageranks, eigenvectors, constraint):

    if len(dist_type) > 0:
        dist_type = '_' + dist_type
    

    for i in range(len(G.degree())):

        name = G.vs[i]['name']

        if name not in vertice_attributes:
            vertice_attributes[name] = {}


        vertice_attributes[name]['clustering'  +  dist_type]   = clusterings[i]
        vertice_attributes[name]['pagerank'    +  dist_type]   = pageranks[i]
        vertice_attributes[name]['eigenvector' +  dist_type]   = eigenvectors[i]

        vertice_attributes[name]['betweenness' +  dist_type]   = betweennesses[i]
        vertice_attributes[name]['closeness'   +  dist_type]   = closenesses[i]
        vertice_attributes[name]['constraint'  +  dist_type]   = constraint[i]


        if dist_type == '':
            vertice_attributes[name]['degree'      +  dist_type]   = degrees[i]
            vertice_attributes[name]['egosize'     +  dist_type]   = neighborhood_sizes[i]




def add_geo_measures(vertice_attributes, G, neighborhoods, dist_type, outfolder, tipus):


    t1 = time.time()

    dist_type_= dist_type
    if len(dist_type) > 0:
        dist_type_ = '_' + dist_type
    

    for i in range(len(G.degree())):

        node      = G.vs[i]
        name      = node['name']
        neighbors = neighborhoods[i]


        vertice_attributes[name]['geo_stdev_of_ego'   +  dist_type_]  =  geo_stdev_of_ego(G, node, neighbors, dist_type)
        vertice_attributes[name]['geo_social_stretch' +  dist_type_]  =  social_stretch(G,   node, neighbors, dist_type)


        if 'geo_size_of_ego' not in vertice_attributes[name]:
            vertice_attributes[name]['geo_size_of_ego'] =  geo_size_of_ego(G, node, neighbors)


    print tipus + '  geo measures - ' +  dist_type_, '\t', round(time.time() - t1,2), ' seconds' 




def calc_network_centralities(G, outfolder, city, infile, tipus, geo, weighted, venue, thresh = ''):

  
    filename           = outfolder + 'networks/' + city  + '_' + tipus + '_' +  thresh  + '_networkmeasures.csv'  
    vertice_attributes = {}
    all_users          = set([line.strip().split('\t')[0] for line in open(infile)])


    t1                 = time.time()
    degrees            = G.degree() 
    neighborhoods      = G.neighborhood(vertices=None, order=1)  
    neighborhood_sizes = G.neighborhood_size(vertices=None, order=1)  


    
    print '\nCentralities, threshold: ', thresh
    print 'Degrees: ', time.time() - t1

    betweennesses, closenesses, clusterings, strengthes, pageranks, eigenvectors, constraint  = get_distance_measures_network(G, None, tipus)
    populate_node_attributes(vertice_attributes, G, '',               degrees, neighborhood_sizes,  betweennesses,      closenesses,      clusterings,      strengthes,      pageranks,      eigenvectors,      constraint)

 
    if geo: 
        betweennesses_inv,  closenesses_inv,  clusterings_inv,  strengthes_inv,  pageranks_inv,  eigenvectors_inv,  constraint_inv  = get_distance_measures_network(G, 'inv_distances',  tipus)
        betweennesses_grav, closenesses_grav, clusterings_grav, strengthes_grav, pageranks_grav, eigenvectors_grav, constraint_grav = get_distance_measures_network(G, 'grav_distances', tipus)
        betweennesses_exp,  closenesses_exp,  clusterings_exp,  strengthes_exp,  pageranks_exp,  eigenvectors_exp,  constraint_exp  = get_distance_measures_network(G, 'exp_distances',  tipus)

        populate_node_attributes(vertice_attributes, G, 'inv_distances',  degrees, neighborhood_sizes,  betweennesses_inv,  closenesses_inv,  clusterings_inv,  strengthes_inv,  pageranks_inv,  eigenvectors_inv,  constraint_inv)
        populate_node_attributes(vertice_attributes, G, 'grav_distances', degrees, neighborhood_sizes,  betweennesses_grav, closenesses_grav, clusterings_grav, strengthes_grav, pageranks_grav, eigenvectors_grav, constraint_grav)
        populate_node_attributes(vertice_attributes, G, 'exp_distances',  degrees, neighborhood_sizes,  betweennesses_exp,  closenesses_exp,  clusterings_exp,  strengthes_exp,  pageranks_exp,  eigenvectors_exp,  constraint_exp)
        

        add_geo_measures(vertice_attributes, G, neighborhoods, 'distances',      outfolder, tipus)
        add_geo_measures(vertice_attributes, G, neighborhoods, 'inv_distances',  outfolder, tipus)
        add_geo_measures(vertice_attributes, G, neighborhoods, 'grav_distances', outfolder, tipus)
        add_geo_measures(vertice_attributes, G, neighborhoods, 'exp_distances',  outfolder, tipus)

    
    if weighted: 


        if '_w' in tipus:
            betweennesses_w, closenesses_w, clusterings_w, strengthes_w, pageranks_w, eigenvectors_w, constraint_w  = get_distance_measures_network(G, 'weight', tipus)

            www = 'w'
            if '_wa' in tipus:
                www = 'wa'

            populate_node_attributes(vertice_attributes, G, www,  degrees, neighborhood_sizes, betweennesses_w, closenesses_w, clusterings_w, strengthes_w, pageranks_w, eigenvectors_w, constraint_w)

        else:

            betweennesses_w,   closenesses_w,   clusterings_w,   strengthes_w,   pageranks_w,   eigenvectors_w,  constraint_w   = get_distance_measures_network(G, 'weight', tipus)
            betweennesses_wa,  closenesses_wa,  clusterings_wa,  strengthes_wa,  pageranks_wa,  eigenvectors_wa, constraint_wa  = get_distance_measures_network(G, 'weight_a', tipus)

            populate_node_attributes(vertice_attributes, G, 'w',   degrees, neighborhood_sizes, betweennesses_w,  closenesses_w,  clusterings_w,  strengthes_w,  pageranks_w,  eigenvectors_w,  constraint_w)
            populate_node_attributes(vertice_attributes, G, 'wa',  degrees, neighborhood_sizes, betweennesses_wa, closenesses_wa, clusterings_wa, strengthes_wa, pageranks_wa, eigenvectors_wa, constraint_wa)



    df = pd.DataFrame.from_dict(vertice_attributes, orient = 'index')
    df.to_csv(filename, na_rep='nan')

    
    return 0











def get_weight_distr(outfolder, outname):

    weights = []



    for w in open(outroot + 'figures/figures/weight_distribution_' + outname + '.dat'):
        weights.append(float(w.strip()))
   


    plt.title('Weight distribution of ' + outname)
    plt.hist(weights, bins = 30, alpha = 0.8)

    #plt.xscale('log')
    plt.yscale('log')
    plt.savefig(outroot + 'figures/network_data/weight_distribution_' + outname + '.png')
    #plt.show()
    





if __name__ == '__main__': 


    city = sys.argv[1]
    #city      = 'bristol'
    eps       = 0.01
    mins      = 3
    LIMIT_num = 0
    outroot   = '../ProcessedData/' + city + '/'
    infile    = outroot + '/user_homes/centroids_filtered/' + city + '_user_homes_dbscan_' + str(eps) + '_' + str(mins) + '_' + str(LIMIT_num) + '_filtered.dat'


    create_folder(outroot + 'networks')
    create_folder(outroot + 'networks/gephi')
    create_folder(outroot + 'figures/network_data')
    



    








    inputs = ParseInput.get_inputs()
    bbox   = inputs[city]
    #do_all_the_networks(city, outroot, infile, bbox)


    if len(sys.argv) == 3:


        if sys.argv[2] == 'friend':

        
            print 'FRIENDS:  Create friendship network' 
            G_friends = get_user_user_friendship_network_igraph(city, outroot, infile)    

            print 'FRIENDS:  Creating gephi files...'
            get_gephi_new(G_friends, outroot, city + '_friendship')   

            print 'FRIENDS:  Calc centrality measures...'
            calc_network_centralities(G_friends, outroot, city, infile, 'friend',       geo = True,  weighted = False, venue = False)

            print 'FRIENDS:  Creating network stats...'
            get_network_stats(G_friends, city, outroot, '_friendship')
            


        elif sys.argv[2] == 'user':


            print 'Create users network' 
 #           G_users   = get_user_user_similarity_network_igraph(city, outroot, infile)
 #           get_gephi_new(G_users, outroot, city + '_users_sim')      
            #for nc_threshold in [5000, 3000, 2000, 1500, 1000, 500, 100]:

            for nc_threshold in [5000, 1000, 500]:
                print 'USERS  THRESHOLD :  ', nc_threshold

                transform_gephi_to_backbone(outroot, city + '_users_sim', nc_threshold)        

                            


                G_users_NC_inv  = create_igraphnw_from_backbone(outroot, city + '_users_sim', 'NC', infile, 'inv',  thresh = str(nc_threshold))
                G_users_NC_grav = create_igraphnw_from_backbone(outroot, city + '_users_sim', 'NC', infile, 'grav', thresh = str(nc_threshold))
                G_users_NC_exp  = create_igraphnw_from_backbone(outroot, city + '_users_sim', 'NC', infile, 'exp',  thresh = str(nc_threshold))
                G_users_NC_w    = create_igraphnw_from_backbone(outroot, city + '_users_sim', 'NC', infile, 'w',    thresh = str(nc_threshold))
                G_users_NC_wa   = create_igraphnw_from_backbone(outroot, city + '_users_sim', 'NC', infile, 'wa',   thresh = str(nc_threshold))

    


                calc_network_centralities(G_users_NC_inv,   outroot, city, infile, 'users_sim_' + 'NC_inv'  ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_users_NC_grav,  outroot, city, infile, 'users_sim_' + 'NC_grav' ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_users_NC_exp,   outroot, city, infile, 'users_sim_' + 'NC_exp'  ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_users_NC_w,     outroot, city, infile, 'users_sim_' + 'NC_w'    ,   geo = True,  weighted = True,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_users_NC_wa,    outroot, city, infile, 'users_sim_' + 'NC_wa'   ,   geo = True,  weighted = True,  venue = False, thresh = str(nc_threshold))           

                #G_users_DF = create_igraphnw_from_backbone(outroot, city + '_users_similarity', 'DF', infile)
                #calc_network_centralities(G_users_DF,   outroot, city, infile, 'users_sim_geo_' + 'DF' ,   geo = True,  weighted = True,  venue = False)
            


    
        elif sys.argv[2] == 'venue':

            

            print 'Create venues network' 
#            G_venues  = get_venue_venue_similarity_network_igraph(city, outroot, infile, bbox)
            print 'Creating gephi files...'
#            get_gephi_new(G_venues,  outroot, city + '_venues_sim')

            #get_distances_between_nodes(outroot, 'venues_sim', city)
            #print 'Creating network stats...'
            #get_network_stats(G_venues,  city, outroot, '_venues_similarity')
            #print 'Calc centrality measures...'
            #calc_network_centralities(G_venues,  outroot, city, infile, 'venues_sim_geo',  geo = True,  weighted = True,  venue = True)
            #get_weight_distr(outroot, city + '_venues_similarity')

            #for nc_threshold in [5000, 3000, 2000, 1500, 1000, 500, 100]:
            for nc_threshold in [5000,1000,500]:#1000, 500, 250, 100, 25, 10, 1]:

                print 'VENUES  THRESHOLD :  ', nc_threshold
                transform_gephi_to_backbone(outroot, city + '_venues_sim', nc_threshold)

                G_venues_NC_inv  = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_sim', 'NC', 'inv',  infile, thresh = str(nc_threshold))
                G_venues_NC_grav = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_sim', 'NC', 'grav', infile, thresh = str(nc_threshold))
                G_venues_NC_exp  = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_sim', 'NC', 'exp',  infile, thresh = str(nc_threshold))
                G_venues_NC_w    = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_sim', 'NC', 'w',    infile, thresh = str(nc_threshold))
                G_venues_NC_wa   = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_sim', 'NC', 'wa',   infile, thresh = str(nc_threshold))

                calc_network_centralities(G_venues_NC_inv,  outroot, city, infile, 'venues_sim_' + 'NC_inv'  ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_venues_NC_grav, outroot, city, infile, 'venues_sim_' + 'NC_grav' ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_venues_NC_exp,  outroot, city, infile, 'venues_sim_' + 'NC_exp'  ,   geo = True,  weighted = False,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_venues_NC_w,    outroot, city, infile, 'venues_sim_' + 'NC_w'    ,   geo = True,  weighted = True,  venue = False, thresh = str(nc_threshold))
                calc_network_centralities(G_venues_NC_wa,   outroot, city, infile, 'venues_sim_' + 'NC_wa'   ,   geo = True,  weighted = True,  venue = False, thresh = str(nc_threshold))           

                #G_venues_DF = create_igraphnw_from_backbone_for_venues(outroot, city + '_venues_similarity', 'DF', infile)
                #calc_network_centralities(G_venues_DF,   outroot, city, infile, 'venues_similarity_' + 'DF' ,   geo = True,  weighted = True,  venue = False)
           



      

        






##   source /opt/virtualenv-python2.7/bin/activate








 




 




 











