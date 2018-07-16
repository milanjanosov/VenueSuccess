import os


tipus = 'venues'

city       = 'london'
outfolder  = '../ProcessedData/' + city + '/'
outname    = '_' + tipus + '_similarity'
foutfolder = outfolder + 'networks/gephi/'  #backboneformat_' + outname + '.dat'
files      = [foutfolder + '/' + fff for fff in os.listdir(foutfolder) if 'NC_B' in fff]




edges_o = len([line for line in open(outfolder  + 'networks/gephi/' + city + '_' + tipus + '_similarity_edges.dat')])
nodes_o =  len([line for line in open(outfolder + 'networks/gephi/' + city + '_' + tipus + '_similarity_nodes.dat')])


print 'original\t', nodes_o, '\t', edges_o




for fn in files:

    threshold = fn.split('_')[2]

    nedges = 0
    nnodes = 0

    n = []

    for line in open(fn):
        nedges += 1
        n += line.strip().split('\t')[0:2]


    nnodes = len(set(n))

    print threshold, '\t', nnodes, '\t', nedges





