import os
import numpy
import sys
import pandas as pd













city      = 'london'
outfolder = '../ProcessedData/' + city + '/' 




relevant_venues = set([line.strip().split('\t')[0] for line in open(outfolder + 'venues_info/venues_ward_full.dat') if 'venue' not in line])

print 'VENUES: ', len(relevant_venues)




files = [ outfolder + 'venues_info/' + city +  ftype for ftype in    ['_liked_venues_stats.dat', '_photoed_venues_stats.dat', '_tipped_venues_stats.dat'] ]


venues_success = {}

for fn in files:


    for ind, line in enumerate(open(fn)):

        if ind % 1000 == 0: 
            print ind

        #if ind == 10: break
        fields  = line.strip().replace('{', '').replace('}','').split('\t')#[1].split(', ')
        venue   = fields[0]

        if venue in relevant_venues:

            success = [fff.split(': ') for fff in   fields[1].split(', ')]

        
            venues_success[venue] = {}

            for s in success:
                meas = s[0].replace('\'','')
                val  = float(s[1])
                venues_success[venue][meas] = val







df = pd.DataFrame.from_dict(venues_success, orient = 'index')
df.to_csv(outfolder + 'venues_info/' + city + '_venues_success_measures.csv' , sep = '\t')



