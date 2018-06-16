import os


city        = 'bristol'
outroot     = '../ProcessedData/' + city 

users_friend_geo   = outroot + '/networks/' + city + '_users_geo_networkmeasures.csv'
users_friend_topo  = outroot + '/networks/' + city + '_users_topo_networkmeasures.csv'
users_sim          = outroot + '/networks/' + city + '_users_topo_networkmeasures.csv'
venues_sim         = outroot + '/networks/' + city + '_venues_sim_geo_networkmeasures.csv'

venues_user = outroot + '/venues_info/' + city + '_venues_users.dat'



venues_users = {}

for line in open(venues_user):
    field = line.strip().split('\t')    
    venue = field[0]
    users = field[1:]

    venues_users[venue] = users




1. függvény ami beolvassa az egyes user featureket és aggregálja másik függvények szerint userekre
