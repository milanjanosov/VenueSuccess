import bs4
import os
import random
import re
import sys
import time
import csv
import urllib2
import sys
import gzip
reload(sys)  
sys.setdefaultencoding('utf8')



def sleep_a_bit(sleeptime, city):
  
    print '4sq ' + city + '  -  sleeping for ' + str(sleeptime) + ' seconds...'       
    for i in xrange(sleeptime,0,-1):
        time.sleep(1)
   
            

def scrape_and_save_url(url, filename, v, sleeptime, city):

    
    try:
    #if 2.0 == 2.0:
           
        if not os.path.exists(filename + '.gz'):
            

            #fout = open('indicies.dat', 'a')
            #fout.write(v+ '\n')
            #fout.close()

            print url

            opener = urllib2.build_opener()
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 5.1; rv:10.0.1) Gecko/20100101 Firefox/10.0.1',}
            opener.addheaders = headers.items()
            html = opener.open(url).read()
                
            with gzip.open(filename + '.gz', 'wb') as f:
                f.write(html)
            
            print "Successfully scraped\t{0} ".format(url)
            sleep_a_bit(4, city)
            
        else:
            print 'Already got it'    
        
    
    except:
        print 'Error sleeping'
        sleep_a_bit(5, city)    


    #    print "Could not connect to\t {0} ".format(url)
     #   f = open("error_log.txt", "a")
     #   sleep_a_bit(5)
     #   f.write(url+'\n')
     #   f.close() 
    



city   = sys.argv[1] # 'london'
folder = '../ProcessedData/' + city + '/venues_info/'
#folder = ''#'../ProcessedData/' + city + '/venues_info/'

#venues = list([line.strip().split('\t')[0] for line in open(folder + city + '_venues_success_measures.csv') if 'tipC' not in line])
#venues = [line.strip() for line in open('indicies.dat')]

venues = [line.strip().split('\t')[0] for line in open('../ProcessedData/'+city+'/timeseries/senior_timeseries_4_13.dat')]


outfolder = 'html/' + city
#outfolder = '../Data/london/html'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)


nnn = len(venues)


#fout = open('indicies.dat', 'w')
#fout.close()

for ind, v in enumerate(reversed(venues)):

    print ind+1, '/', nnn, '\t', 

    filename = outfolder + '/' + v + '_html.dat'     
    #if not os.path.exists(filename):
    url = 'https://foursquare.com/v/' +  v 


    scrape_and_save_url(url, filename, v, 0, city)   










