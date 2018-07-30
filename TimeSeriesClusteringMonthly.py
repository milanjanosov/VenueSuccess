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




def getBinnedDistribution(x, y, nbins):

    n, bins   = np.histogram(x, bins=nbins)
    sy, _  = np.histogram(x, bins=nbins, weights=y)
    #sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy/n
     
    #std = np.sqrt(sy2/n - mean*mean) 

    return _, mean



series = []
tseries = []

for ind, line in enumerate(open('TIMESERIES4.DAT')):
    #if ind == 50: break
    series.append(  np.asarray([float(fff) for fff in line.strip().split('\t')]))


for ind, line in enumerate(open('TIMESERIES_tims.dat')):
    #if ind == 50: break
    tseries.append(  np.asarray([float(fff) for fff in line.strip().split('\t')]))






print (len(series))

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


    #if i == 2: break

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
    



    cnt = [(c, len(n)) for (c, n) in cluster_dict.items()]
    num = min(len(cnt), 5)
    cnt = sorted(cnt, key=lambda tup: tup[1], reverse = True)[0:num]


    ts1.append((i, cnt[0][1]))
    if len(cnt) > 1: ts2.append((i, cnt[1][1]))
    if len(cnt) > 2: ts3.append((i, cnt[2][1]))
    if len(cnt) > 3: ts4.append((i, cnt[3][1]))
    if len(cnt) > 4: ts5.append((i, cnt[4][1]))


    if len(cnt) > 3:
        print (i, '\t', nc, nnodes, '\t', cnt[0:2])

    
    top5cluster = [c[0] for c in cnt]

   # print (i, '\t', nc, nnodes)#, '\t', cluster_dict, '\n')





    print (top5cluster, len(top5cluster))


    indicies = [(0,0), (0,1), (1,0), (0,2), (1,1)]

    #if len(cnt) > 3:
    if nc == 14 and nnodes == 2985:
   # if cnt[0][0] < 2 * cnt[1][0]:


        f, ax = plt.subplots(2, 3, figsize=(15, 20))

        ind = 0

        for c, nodes in cluster_dict.items():

            if c in top5cluster:




                sss  = []
                tsss = []
                subseries = []
                for n in nodes:
                    #ax[indicies[ind]].plot(series[int(n)], linewidth = 0.4, color = 'grey', alpha = 0.25)
                    subseries.append(series[int(n)])
                    sss  += list(series[int(n)])
                    tsss += list(tseries[int(n)])


                for n in nodes:
                    alpha_ = 0.9 
                    if len(subseries) > 100:
                        alpha_ = 0.35
                    if len(subseries) > 450:
                        alpha_ = 0.15
                    
                    ax[indicies[ind]].plot(tseries[int(n)], series[int(n)], linewidth = 0.4, color = 'grey', alpha = alpha_)




                ax[indicies[ind]].set_title('Number of venues = ' +  str(len(subseries)), fontsize = 19)


                MINS = min(tsss)
                MAXS = max(tsss)
                N    = 20
            
                bins = np.arange(MINS, MAXS, N)
                avgpoints= {}

                #print(len(sss), type(sss), len(tsss), type(tsss)) 
                
                bx, by = getBinnedDistribution(tsss, sss, 15)
                bx= (bx[1:] + bx[:-1])/2

                ax[indicies[ind]].plot(bx, by, linewidth = 3, color = 'r')
    
                ind += 1
   


    #if i > 1200:
    num_leaves.append(nnodes)
    num_clusters.append(nc)
    


plt.show()


'''

x,y = zip(*ts1)
plt.plot(x,y, 'r-', color = 'steelblue', markersize = 6, alpha = 0.95, label = '#1 cluster')

x,y = zip(*ts2)
plt.plot(x,y, 'r-', color = 'darkred', markersize = 6, alpha = 0.95, label = '#2 cluster')

x,y = zip(*ts3)
plt.plot(x,y, 'r-', color = 'grey', markersize = 6, alpha = 0.95, label = '#3 cluster')

x,y = zip(*ts4)
plt.plot(x,y, 'r-', color = 'darkorange', markersize = 6, alpha = 0.95, label = '#4 cluster')

x,y = zip(*ts5)
plt.plot(x,y, 'r-', color = 'darkgreen', markersize = 6, alpha = 0.95, label = '#5 cluster')



#plt.plot(num_leaves, num_clusters, 'o', color = 'steelblue', markersize = 6, alpha = 0.7)
#plt.xlabel('Number of clustered nodes', fontsize = 17)
#plt.ylabel('Number of clusters', fontsize = 17)

#model3.plot('a.png')
#plt.yscale('log')
plt.xlim([1350, 1450])
plt.title('Top5 largest clusters size', fontsize = 20)
plt.xlabel('Iteration', fontsize = 17)
plt.ylabel('Cluster size', fontsize = 17)
plt.tight_layout()
plt.show()

'''



