from dtaidistance import dtw
from dtaidistance import clustering
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from collections import Counter
from dtaidistance import clustering




from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import savgol_filter



def transform_ts(x, maxTT):

    minx = min(x)  
    x    = [xx - minx for xx in x] 
    maxx = max(x)
    x    = [xx * maxTT / maxx for xx in x]
    
    return x


def getBinnedDistribution(x, y, nbins):

    n, bins = np.histogram(x, bins=nbins)
    sy, _   = np.histogram(x, bins=nbins, weights=y)
    mean    = sy/n

    return _, mean



def avg_curve(a,b):
    
    return np.asarray([(a[i] + b[i])/2.0 for i in range(len(a))])




series = []

for ind, line in enumerate(open('TIMESERIES_911.DAT')):
    #if ind == 500: break
    series.append(  savgol_filter(np.asarray([float(fff) for fff in line.strip().split('\t')]), 7, 3)   )
    #series.append( np.asarray([float(fff) for fff in line.strip().split('\t')])   )


dists = dtw.distance_matrix_fast(series)




# model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
# Augment Hierarchical object to keep track of the full tree
# model2 = clustering.HierarchicalTree(model1)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
cluster_idx = model3.fit(series)
#print (dir(model3))
#print (model3.linkage)






linkage_matrix = model3.linkage

n = len(series)
#print(n)
cluster_dict = dict()



num_leaves   = []
num_clusters = []



ts1 = []
ts2 = []
ts3 = []
ts4 = []
ts5 = []



for i in range(0, n-1):

    new_cluster_id = n+i
    old_cluster_id_0 = linkage_matrix[i, 0]
    old_cluster_id_1 = linkage_matrix[i, 1]
    combined_ids = list()
    if old_cluster_id_0 in cluster_dict:
        combined_ids += cluster_dict[old_cluster_id_0]
        del cluster_dict[old_cluster_id_0]
    else:
        combined_ids += [old_cluster_id_0]
    if old_cluster_id_1 in cluster_dict:
        combined_ids += cluster_dict[old_cluster_id_1]
        del cluster_dict[old_cluster_id_1]
    else:
        combined_ids += [old_cluster_id_1]
    cluster_dict[new_cluster_id] = combined_ids

    nodes_included = []
    for v in cluster_dict.values():
        nodes_included += v


    nc     = len(cluster_dict)  
    nnodes = len(set(nodes_included))
    


    NNN = 10   # 5 # 6 # 10


    for comparison in [2, 5, 10, 15]:
        #comparison = 1


        cnt = [(c, len(n)) for (c, n) in cluster_dict.items()]
        num = min(len(cnt), NNN)
        cnt = sorted(cnt, key=lambda tup: tup[1], reverse = True)[0:num]



        biggest = sum([cc[1] for cc in cnt])

        if len(cnt) > 2:
            print (np.std( [c[1] for c in cnt]), '\t', biggest, '\t', nc, nnodes, '\t', [cc[1] for cc in cnt])

        
        top5cluster = [c[0] for c in cnt]

       # print (i, '\t', nc, nnodes)#, '\t', cluster_dict, '\n')



        indicies =[(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), (1,5)]

        if not os.path.exists('clusters_warped/' + str(NNN)):
            os.makedirs('clusters_warped/' + str(NNN))
     
        if  biggest > 2000:





        #if biggest > 350:

            f, ax = plt.subplots(2, 5, figsize=(20, 8))   # 20, 8

            ind = 0

            for c, nodes in cluster_dict.items():

        


                if c in top5cluster:

                
                    ttt = []
                    sss = []

                    subseries     = []
                    subseries_map = []




                    x = series[comparison]

                    xallm = []
                    tallm = []


                    for n in nodes:

                        if n < 3687:



                            y  = series[int(n)]

                            map_x, map_y = list(zip(*dtw.warping_path(x, y)))

                            map_x = np.asarray(map_x)
                            map_y = np.asarray(map_y)

                            if len(y):
                                maxl = len(y)




                            xallm += list(y[map_y])
                            tallm += range(len(map_y))

                            if i == 1:
                                maxl  = len(x)
                                map_x = np.asarray(map_x)
                                xallm = list(x[map_x]) 
                                tallm = list(range(len(map_x))) 
                        
                                subseries_map.append(x[map_x])
                                subseries.append(x)



                            subseries.append(y)
                            subseries_map.append(y[map_y])

                            sss  += list(series[int(n)])
                            ttt  += transform_ts(list(range(len(series[int(n)]))), 11)

              

                    if ind < 5:

                        for ijk, linetoplot in enumerate(subseries):

                            ax[indicies[ind]].plot(linetoplot,           linewidth = 0.4, color = 'grey', alpha = 0.15)
                            ax[indicies[ind+5]].plot(subseries_map[ijk], linewidth = 0.4, color = 'grey', alpha = 0.15)


                        ax[indicies[ind]].set_title('Number of venues = ' +  str(len(subseries)), fontsize = 15)






                    bx, by = getBinnedDistribution(ttt, sss, 8  )
                    bx = (bx[1:] + bx[:-1])/2
                    if ind < 5: ax[indicies[ind]].plot(bx, by, linewidth = 3, color = 'r')







                    fout = open('clusters_warped/' + str(NNN) + '/ref_' + str(biggest) + '_' + str(ind) + '.dat', 'w')
                    fout.write('\t'.join([str(b) for b in bx]) + '\n')
                    fout.write('\t'.join([str(b) for b in by]) + '\n')
                    fout.close()






                    bx, by = getBinnedDistribution(tallm, xallm, 8)    
                    bx= (bx[1:] + bx[:-1])/2
                    if ind < 5: ax[indicies[ind+5]].plot(bx, by, 'r', linewidth = 3)


                    fout = open('clusters_warped/' + str(NNN) + '/comp_' + str(comparison) + '_' + str(biggest) + '_' + str(ind) + '.dat', 'w')
                    fout.write('\t'.join([str(b) for b in bx]) + '\n')
                    fout.write('\t'.join([str(b) for b in by]) + '\n')
                    fout.close()






                    ind += 1
             

             
            #plt.show() 
            plt.savefig('clusters_warped/'+str(NNN)+'/' + str(biggest) + '_' + str(comparison) + '.png')    
            plt.close()

        #if i > 1200:
        num_leaves.append(nnodes)
        num_clusters.append(nc)
        


 



