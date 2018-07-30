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






def cluster_the_ts_curves(infile, outfolder, maturity):

    series   = {}
    venues   = []
    indicies = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), (1,5)]

    for ind, line in enumerate(open(infile)):
        fields = line.strip().split('\t')
        venue  = fields[0]
        ts     = fields[1:]
        venues.append(venue)
        #if ind == 500: break
        series[venue] = savgol_filter(np.asarray([float(fff) for fff in ts  ] ), 7, 3)   



    dists          = dtw.distance_matrix_fast(list(series.values()))
    model3         = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx    = model3.fit(list(series.values()))
    linkage_matrix = model3.linkage

    nnn = len(series)
    cluster_dict = {}


    if not os.path.exists(maturity):
        os.makedirs(maturity)



    for i in range(0, nnn-1):

        new_cluster_id = nnn+i
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
            if max(v) > nnn:
                print ('FAASZ  ', nnn, max(v))


        nc     = len(cluster_dict)  
        nnodes = len(set(nodes_included))
        

        #for NNN in [6]:
        #for NNN in [3, 5, 6, 10]:
        for NNN in [3]:
    
            #NNN = 6   # 5 # 6 # 10

            figfolder   = outfolder + '/' + maturity + '/figs_clusters/'   + str(NNN)
            curvefodler = outfolder + '/' + maturity + '/avg_curves/'      + str(NNN)
            vensfolder  = outfolder + '/' + maturity + '/clusters_venues/' + str(NNN)

            if not os.path.exists(figfolder):    os.makedirs(figfolder)
            if not os.path.exists(curvefodler):  os.makedirs(curvefodler)
            if not os.path.exists(vensfolder):   os.makedirs(vensfolder)


            cnt = [(c, len(n)) for (c, n) in cluster_dict.items()]
            num = min(len(cnt), NNN)
            cnt = sorted(cnt, key=lambda tup: tup[1], reverse = True)[0:num]

            biggest     = sum([cc[1] for cc in cnt])   
            top5cluster = [c[0] for c in cnt]




            if biggest > len(series) / 2:

                f, ax = plt.subplots(2, 5, figsize=(20, 8)) 
                ind   = 0

                for ccc, nodes in cluster_dict.items():
                    
                    if ccc in top5cluster:
      
                        ttt = []
                        sss = []

                        cluster_vens = []
                        subseries    = []

                        for n in nodes:

                            subseries.append(list(series.values())[int(n)])
                          
                            sss  += list(list(series.values())[int(n)])
                            ttt  += transform_ts(list(range(len(list(series.values())[int(n)]))), 11)
                     
                        for n in nodes:

                            cluster_vens.append( list(series.keys())[int(n)] )
                            linetotplot = list(series.values())[int(n)]
                            ax[indicies[ind]].plot(linetotplot, linewidth = 0.4, color = 'grey', alpha = 0.15)



                        ffout = open(vensfolder + '/venues_in_' + str(ind) + '_' + str(biggest) + '.dat', 'w')
                        ffout.write( '\n'.join(cluster_vens))
                        ffout.close()
            


                        ax[indicies[ind]].set_title('Number of venues = ' +  str(len(subseries)), fontsize = 15)
                

                        bx, by = getBinnedDistribution(ttt, sss, 8  )
                        bx     = (bx[1:] + bx[:-1])/2


                        fout = open(curvefodler + '/avg_curve_' + str(ind) + '_' + str(biggest) + '.dat', 'w')
                        fout.write('\t'.join([str(b) for b in bx]) + '\n')
                        fout.write('\t'.join([str(b) for b in by]) + '\n')
                        fout.close()
                        ax[indicies[ind]].plot(bx, by, linewidth = 3, color = 'r')



                        ind += 1
                        






            
                  
                plt.savefig(figfolder +'/top_' + str(NNN) + '_clusters_' + str(biggest) + '.png')    
                plt.close()
        


            


if __name__ == "__main__":


    city       = 'london'
    outfolder  = '../ProcessedData/' + city + '/timeseries/'
    infile     = outfolder + 'senior_timeseries_8_12.dat'
    #infile     = outfolder + 'mid_timeseries_4_9.dat'

    cluster_the_ts_curves(infile, outfolder, 'senior_8_12')









