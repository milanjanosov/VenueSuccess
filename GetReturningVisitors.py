from collections import Counter


### the relevant guys
city = 'london'
outroot   = '../ProcessedData/' + city + '/venues_info/'

relevant_venues = list(set([line.strip().split('\t')[0] for line in open(outroot + '/venues_ward_full.dat') if 'venue' not in line]))



''' the number of diff users a venue has ''' 

venues_users_l = {}
print 'Number of distinct users...'
for ind, line in enumerate( open(outroot + 'venues_photod_categories_times.dat') ) :

    if ind == 10: break
    user, venue, a, b, c = line.strip().split('\t')
    
    if venue in relevant_venues:    
        if venue not in venues_users_l:
            venues_users_l[venue] = [user]
        else:
            venues_users_l[venue].append(user)
        
venues_users = {}
for v, u in venues_users_l.items():
    venues_users[v] = len(set(u))



records = [ (line.strip().split('\t')[0] + '_' + line.strip().split('\t')[1])  for line in open(outroot + 'venues_photod_categories_times.dat')    ]
records_cnt = dict(Counter(records))





''' the number of regular users a venue has ''' 

venues_regulars_l = {}
print 'Number of regular users...'
for ind, (venue_user, cnt) in enumerate(records_cnt.items()):

    if ind == 10: break
    if venue in relevant_venues:    
        print venue
        if cnt > 1 :   
            user, venue = venue_user.split('_')
            if venue not in venues_regulars_l:
                venues_regulars_l[venue] = [user]
            else:
                venues_regulars_l[venue].append(user)
            
venues_regulars = {}
for v, u in venues_regulars_l.items():
    venues_regulars[v] = len(set(u)) 
    




''' writing out the result '''
fout = open(outroot + 'venues_REGULARS.dat', 'w')

for venue, usercn in venues_users.items():

    regulars = 0
    if venue in venues_regulars:
        regulars = venues_regulars[venue]

    fout.write(venue + '\t' + str(usercn) + '\t' + str(regulars) + '\t' + str(float(regulars) / float(usercn) ) + '\n')


fout.close()

    





