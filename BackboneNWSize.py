import os

city       = 'london'
outfolder  = '../ProcessedData/' + city + '/'
outname    = '_venues_similarity'
foutfolder = outfolder + 'networks/gephi/'  #backboneformat_' + outname + '.dat'
files      = [foutfolder + '/' + fff for fff in os.listdir(foutfolder) if '_NC' in fff]


#files = ['NC_BACKBONE_250_london_venues_similarity_edges.dat']


for fn in files:

    threshold = fn, fn.split('_')[2]

    nedges = 0
    nnodes = 0

    n = []

    for line in open(fn):
        nedges += 1
        n += line.strip().split('\t')[0:2]


    nnodes = len(set(n))

    print threshold, '\t', nnodes, '\t', nnedges
