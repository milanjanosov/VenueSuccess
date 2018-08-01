import os
import gzip
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd




def get_venues_money(city, outfolder):

  
    files = ['html/' + fn for fn in os.listdir('html')]

    venues_money= {}
    nnn = len(files)

    for ind, fn in enumerate(files):

        #if ind == 100: break
        print 'html parsing:   ', ind, '/', nnn


        if 'dat.gz' == fn.split('.', 1)[-1] :

            with gzip.open(fn) as myfile:
            
                html = myfile.read()


                soup = BeautifulSoup(html, "lxml")

                results = soup.findAll('span', {'itemprop' : 'priceRange'})

                pricerange = 'na'

                try:
                     pricerange = len(results[0].text)

                except:
                    pass

                venues_money[fn.split('/')[1].split('_')[0]] = pricerange
                                                



    outfile = outfolder + '/venues_info/' + city + '_venues_4sq_money.dat'

    df = pd.DataFrame.from_dict(venues_money, orient = 'index')
    df = df.rename(columns = { 0: 'money'})
    df.index.name = "venue"
    df.to_csv(outfile, na_rep='nan')








def get_users_venues_money(outfolder, city):


    infile        = outfolder + '/venues_info/' + city + '_venues_4sq_money.dat'
    dfvenues      = pd.read_csv(infile, sep = ',', index_col = 0)
    venues_moneys = dict(zip(dfvenues.index, dfvenues.money))



    users_moneys = {}

    for ind, line in enumerate(open(outfolder + 'user_info/london_user_venues_full_locals_filtered.dat')):


        print 'aggregating users money profile    ', ind

        #if ind == 10000: break

        fields = line.strip().split('\t')
        user   = fields[0]
        venues = [fff.split(',')[0] for fff in fields[1:]]

         
        for venue in venues:
            if venue in venues_moneys:           
                money = venues_moneys[venue]

                if 'na' != money:

                    money = float(money)
    
                    if user not in users_moneys:
                        users_moneys[user] = [money]
                    else:
                        users_moneys[user].append(money)
                


    fout = open(outfolder + '/venues_info/' + city + '_venues_users_moneys.dat', 'w')
    for user, m in users_moneys.items():
        fout.write( user + '\t' + '\t'.join([str(mm) for mm in m]) + '\n')
    fout.close()
    

    

def get_venues_users_moneys(outfolder, city):


    # get the users money lists
    users_moneys = {}
    for ind, line in enumerate(open(outfolder + '/venues_info/' + city + '_venues_users_moneys.dat')):
        print 'Read money    ', ind
        #if ind == 1000: break
        fields = line.strip().split('\t')
        user   = fields[0]
        moneys = [float(fff) for fff in fields[1:]]

       
        users_moneys[user] = {}

        for m in moneys:
            if m not in users_moneys[user]:
                users_moneys[user][m]  = 1
            else:
                users_moneys[user][m] += 1 



    # get the venues users
    venues_users = {}

    relevant_venues = set([line.strip().split('\t')[0] for line in open(outfolder + '/venues_info/venues_ward_full.dat') if 'venue' not in line])

    for ind, line in enumerate(open(outfolder + '/venues_info/london_venues_users.dat')):
        print 'Read venues    ', ind
        #if ind == 100: break
        fields = line.strip().split('\t')
        venue = fields[0]

        if venue in relevant_venues:

            users = fields[1:] 
            venues_users[venue] = users

    print len(venues_users)

    print 'AA', len(venues_users), '\n\n\n'



    # get the money lists for the venues
    venues_users_moneys = {}
    for ind, (venue, users) in enumerate(venues_users.items()):

        venues_users_moneys[venue] = {}


        if ind == 1000: break

        print 'Finalize, venues...    ', ind

        for user in users:
            if user in users_moneys:

                

                for moneyvalue, moneycount in users_moneys[user].items():

                    if moneyvalue not in venues_users_moneys[venue]:
                        venues_users_moneys[venue][moneyvalue]  = moneycount
                    else:
                        venues_users_moneys[venue][moneyvalue] += moneycount



    # venues money_stats as features
    venues_money_stats = {}

    for ind, venue in enumerate(venues_users):


        print 'aggregating venues money profile    ', ind

        #if ind == 100: break

        if venue in venues_users_moneys:
            umoneys = venues_users_moneys[venue]

            N  = float(len(umoneys))
            c1 = umoneys.count(1) / N
            c2 = umoneys.count(2) / N
            c3 = umoneys.count(3) / N
            c4 = umoneys.count(4) / N

            if venue not in venues_money_stats: venues_money_stats[venue] = {}

            venues_money_stats[venue]['m_1_fraction'] = c1
            venues_money_stats[venue]['m_2_fraction'] = c2
            venues_money_stats[venue]['m_3_fraction'] = c3
            venues_money_stats[venue]['m_4_fraction'] = c4

            venues_money_stats[venue]['m_4_avg'] = np.mean(umoneys)
            venues_money_stats[venue]['m_4_std'] = np.std(umoneys) 

        else:

            if venue not in venues_money_stats: venues_money_stats[venue] = {}

            venues_money_stats[venue]['m_1_fraction'] = 0
            venues_money_stats[venue]['m_2_fraction'] = 0
            venues_money_stats[venue]['m_3_fraction'] = 0
            venues_money_stats[venue]['m_4_fraction'] = 0

            venues_money_stats[venue]['m_4_avg'] = 0
            venues_money_stats[venue]['m_4_std'] = 0
        




    outfile = outfolder + '/venues_info/' + city + '_venues_FINAL_money.dat'

    df = pd.DataFrame.from_dict(venues_money_stats, orient = 'index')
    df = df.rename(columns = { 0: 'money'})
    df.index.name = "venue"

    df.to_csv(outfile, na_rep='nan')#, header = True)

   

city        = 'london'
outfolder   = '../ProcessedData/' + city + '/'


#get_venues_money(city, outfolder)
#get_users_venues_money(outfolder, city)
get_venues_users_moneys(outfolder, city)






 
