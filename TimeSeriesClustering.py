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
from scipy.signal import savgol_filter
from dtaidistance import clustering


city       = 'london'
outfolder  = '../ProcessedData/' + city + '/'



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

    cnt = [(c, len(n)) for (c, n) in cluster_dict.items()]
    num = min(len(cnt), NNN)
    cnt = sorted(cnt, key=lambda tup: tup[1], reverse = True)[0:num]


    ts1.append((i, cnt[0][1]))
    if len(cnt) > 1: ts2.append((i, cnt[1][1]))
    if len(cnt) > 2: ts3.append((i, cnt[2][1]))
    #if len(cnt) > 3: ts4.append((i, cnt[3][1]))
    #if len(cnt) > 4: ts5.append((i, cnt[4][1]))

    biggest = sum([cc[1] for cc in cnt])

    if len(cnt) > 2:
        print (np.std( [c[1] for c in cnt]), '\t', biggest, '\t', nc, nnodes, '\t', [cc[1] for cc in cnt])

    
    top5cluster = [c[0] for c in cnt]

   # print (i, '\t', nc, nnodes)#, '\t', cluster_dict, '\n')





    #print (top5cluster, len(top5cluster))


    indicies =[(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), (1,5)]

    if not os.path.exists('clusters/' + str(NNN)):
        os.makedirs('clusters/' + str(NNN))
 
    #if nc == 15 and nnodes == 3687:

    if biggest > 1900:

        f, ax = plt.subplots(2, 5, figsize=(20, 8))   # 20, 8

        ind = 0

        for c, nodes in cluster_dict.items():

    


            if c in top5cluster:

            
                ttt = []
                sss = []

                subseries = []
                for n in nodes:

                    if n < 3687:

                        #ax[indicies[ind]].plot(series[int(n)], linewidth = 0.4, color = 'grey', alpha = 0.25)
                        subseries.append(series[int(n)])

                        sss  += list(series[int(n)])
                        ttt  += transform_ts(list(range(len(series[int(n)]))), 11)

          


                for n in nodes:

                   if n < 3687:

                        alpha_ = 0.5 
                        if len(subseries) > 100:
                            alpha_ = 0.35
                        if len(subseries) > 450:
                            alpha_ = 0.15
                        

                        linetotplot = series[int(n)]

                        ax[indicies[ind]].plot(linetotplot, linewidth = 0.4, color = 'grey', alpha = alpha_)
                        #ax[indicies[ind+3]].plot(series[int(n)], linewidth = 0.4, color = 'grey', alpha = alpha_)

    

                ax[indicies[ind]].set_title('Number of venues = ' +  str(len(subseries)), fontsize = 15)

                c = []
                #for c_i in range(len(subseries[0])):
                #    c.append(np.mean([subseries[r][c_i]   for r in  range(len(subseries))]))
                    #c[c_i] = np.mean(subseries[i][c_i] for i in range(len(subseries))   )

                #c = savgol_filter(c, 7, 3)

                #ax[indicies[ind]].plot(c, linewidth = 3, color = 'r')

                #c = savgol_filter(c, 7, 3)
                #ax[indicies[ind + 3]].plot(c, linewidth = 3, color = 'r')
                

                bx, by = getBinnedDistribution(ttt, sss, 8  )
                bx= (bx[1:] + bx[:-1])/2


                fout = open('clusters/' + str(NNN) + '/' + str(biggest) + '_' + str(ind) + '.dat', 'w')
                fout.write('\t'.join([str(b) for b in bx]) + '\n')
                fout.write('\t'.join([str(b) for b in by]) + '\n')
                fout.close()
                ax[indicies[ind]].plot(bx, by, linewidth = 3, color = 'r')

                ind += 1

                
          
        plt.savefig('clusters/'+str(NNN)+'/' + str(biggest) + '.png')    
        plt.close()

    #if i > 1200:
    num_leaves.append(nnodes)
    num_clusters.append(nc)
    


    plt.show()




x,y = zip(*ts1)
plt.plot(x,y, 'r-', color = 'steelblue', markersize = 6, alpha = 0.95, label = '#1 cluster')

x,y = zip(*ts2)
plt.plot(x,y, 'r-', color = 'darkred', markersize = 6, alpha = 0.95, label = '#2 cluster')

x,y = zip(*ts3)
plt.plot(x,y, 'r-', color = 'grey', markersize = 6, alpha = 0.95, label = '#3 cluster')

'''x,y = zip(*ts4)
plt.plot(x,y, 'r-', color = 'darkorange', markersize = 6, alpha = 0.95, label = '#4 cluster')

x,y = zip(*ts5)
plt.plot(x,y, 'r-', color = 'darkgreen', markersize = 6, alpha = 0.95, label = '#5 cluster')

'''

#plt.plot(num_leaves, num_clusters, 'o', color = 'steelblue', markersize = 6, alpha = 0.7)
#plt.xlabel('Number of clustered nodes', fontsize = 17)
#plt.ylabel('Number of clusters', fontsize = 17)

#model3.plot('a.png')
#plt.yscale('log')
plt.xlim([3500, 3720])
plt.title('Top3 largest clusters size, 5.5y tenure', fontsize = 20)
plt.xlabel('Iteration', fontsize = 17)
plt.ylabel('Cluster size', fontsize = 17)
plt.tight_layout()
plt.show()




