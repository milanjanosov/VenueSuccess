import os
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
import random
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as hac



def getBinnedDistribution(x, y, nbins):

    n, bins = np.histogram(x, bins=nbins)
    sy, _   = np.histogram(x, bins=nbins, weights=y)
    mean    = sy/n

    return _, mean



def transform_ts(x, maxTT):

    minx = min(x)  
    x    = [xx - minx for xx in x] 
    maxx = max(x)
    x    = [xx * maxTT / maxx for xx in x]
    
    return x



def get_curves(outfolder):


    files = []
    for NNN in [3, 5, 6, 10]:
           
        curvefodler  = outfolder + '/avg_curves/' + str(NNN)   
        files       += [curvefodler + '/' + fff for fff in os.listdir(curvefodler) if '.dat' in fff]


    curves = []

    xs = set()
    ys = set()

    fffiles = []

    for fn in files:

        with open(fn) as myfile:
            x, y = myfile.read().strip().split('\n')

            if y not in ys:
                fffiles.append(fn)

            #xs.add(x)
            ys.add(y)

    #x      = [float(xx) for xx in x.split('\t')]
    curves = [[float(yy) for yy in y.split('\t')] for y in list(ys)]
    print(len(curves))
    
    return curves, fffiles
   

   


def get_distance_matrix(curves):

    n = len(curves)


    distances = np.zeros((n,n))

    for i, x in enumerate(curves[0:n]):
        for j, y in enumerate(curves[0:n]):

            if i != j:
                xx = np.asarray(x).reshape(-1, 1)
                yy = np.asarray(y).reshape(-1, 1)

                dist, cost, acc, path = dtw(xx, yy, dist=lambda xx, yy: norm(xx - yy, ord=1))
                map_x, map_y = path

                distances[i,j] = spearmanr(xx[map_x],yy[map_y])[0]

    return distances
                


def plot_avg_curves(curves, outfolder, maturity):
    
    f, ax = plt.subplots(10, 6, figsize=(20, 15))

    print len(curves)

    indicies = [(i,j) for i in range(10) for j in range(6)]

    for ijk in range(len(curves)):

        iijk = ijk
        ax[indicies[ijk]].plot(curves[iijk],  linewidth = 3)
        ax[indicies[ijk]].set_title(str(ijk))
        ax[indicies[ijk]].spines['bottom'].set_color('lightgrey')
        ax[indicies[ijk]].spines['top'].set_color('lightgrey')
        ax[indicies[ijk]].spines['left'].set_color('lightgrey')
        ax[indicies[ijk]].spines['right'].set_color('lightgrey')


    ax[5,5].set_axis_off()

    for i in range(9):
        for j in range(6):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
         
    folder = outfolder + '/avg_curves_clusters/'
    if not os.path.exists(folder):
        os.makedirs(folder) 

    plt.savefig(folder + maturity + '_avg_curves.png')   
    plt.tight_layout() 
    
    





def do_hclustering(curves, method_, outfolder, maturity):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.get_yaxis().set_visible(False)

    Z = hac.linkage(curves, method=method_, metric='correlation')
    d = hac.dendrogram( Z, ax = ax, color_threshold = 1.3,  leaf_rotation=90.,  leaf_font_size=14., show_leaf_counts = True, no_labels = False)

    folder = outfolder + '/avg_curves_clusters/'
    if not os.path.exists(folder):
        os.makedirs(folder) 
    plt.savefig(folder + maturity + '_dendogram_' + method_ + '.png')

    return Z







def plot_clusters(linkage_matrix, outfolder, maturity, method, files):

    Z = linkage_matrix 
    n = len(curves)

    cluster_dict = dict()
    folder       = outfolder + '/avg_curves_clusters/'


    nnn = 5
    mmm = 6
    f, ax = plt.subplots(mmm,nnn, figsize=(16, 10))
    g, bx = plt.subplots(mmm,nnn, figsize=(16, 10))

    indind = 0

    for i in range(0, 57):

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

        n_c = len(cluster_dict.keys())
        n_n = len([item for sublist in cluster_dict.values() for item in sublist])

        print i, n_c, n_n

        if i > 40 and n_c > 3 and n_c < 7: 
        #if n_c > 3 and n_c < 7: 
            
            fffolder = folder + '/' + str(n_c) + '_' + str(n_n)  + '/'
            if not os.path.exists(fffolder): os.makedirs(fffolder)



            for jind, (clus, nodes) in enumerate(cluster_dict.items()):


                ttt = []
                sss = []

                fffout = open(fffolder + '/cluster_' + str(jind) + '.dat', 'w')


                for nnodes in nodes:



                    fffout.write(files[int(nnodes)] + '\n')

                    ax[jind, indind].plot(curves[int(nnodes)])
                    bx[jind, indind].plot(curves[int(nnodes)], linewidth = 0.8, color = 'lightgrey', alpha = 0.7)

                    sss  += curves[int(nnodes)]
                    ttt  += range (len(curves[int(nnodes)] ))

                fffout.close()  

                if jind == 0:
                    ax[jind, indind].set_title('Nodes in = ' + str(n_n) + '\ncluster size = ' + str(len(nodes)), fontsize = 11)
                    bx[jind, indind].set_title('Nodes in = ' + str(n_n) + '\ncluster size = ' + str(len(nodes)), fontsize = 11)
                else:
                    ax[jind, indind].set_title('cluster size = ' + str(len(nodes)), fontsize = 11)
                    bx[jind, indind].set_title('cluster size = ' + str(len(nodes)), fontsize = 11)

                ax[jind, indind].spines['bottom'].set_color('lightgrey')
                ax[jind, indind].spines['top'].set_color('lightgrey')
                ax[jind, indind].spines['left'].set_color('lightgrey')
                ax[jind, indind].spines['right'].set_color('lightgrey')
                ax[jind, indind].get_yaxis().set_visible(False)
                ax[jind, indind].get_xaxis().set_visible(False)

                bx[jind, indind].spines['bottom'].set_color('lightgrey')
                bx[jind, indind].spines['top'].set_color('lightgrey')
                bx[jind, indind].spines['left'].set_color('lightgrey')
                bx[jind, indind].spines['right'].set_color('lightgrey')
                bx[jind, indind].get_yaxis().set_visible(False)
                bx[jind, indind].get_xaxis().set_visible(False)
                

                bbx, by = getBinnedDistribution(ttt, sss, 8  )
                bbx     = (bbx[1:] + bbx[:-1])/2
                bx[jind, indind].plot(bbx, by, linewidth = 3, color = 'r')
    

            indind += 1


    for i in range(mmm):
        for j in range(nnn):
            if len(ax[i,j].lines) == 0:
                ax[i,j].set_axis_off()
                bx[i,j].set_axis_off()

    f.tight_layout()
    f.savefig(folder + maturity + '_clusters_colored_'   + method + '.png')
    
    g.tight_layout()
    g.savefig(folder + maturity + '_clusters_greyscale_' + method + '.png')












maturity   = 'senior_8_12'
method     = 'complete'
city       = 'london'
outfolder  = '../ProcessedData/' + city + '/timeseries/' + maturity

curves, files = get_curves(outfolder)
distances = get_distance_matrix(curves)
plot_avg_curves(curves,  outfolder, maturity)
linkage_matrix = do_hclustering(curves, method, outfolder, maturity)
plot_clusters(linkage_matrix, outfolder, maturity, method, files)

