import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import pandas as pd
from scipy.stats import spearmanr
from collections import Counter
from dtw import dtw
from numpy.linalg import norm





''' =========================================================== '''
'''                         GENERAL HELPERS                     '''
''' =========================================================== '''


def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 
    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())


def round_unix_date(dt_series, seconds=60, up=False):
    return dt_series // seconds * seconds + seconds * up


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





''' =========================================================== '''
'''                   PROCESS THE TIME SERIES                   '''
''' =========================================================== '''


def get_venues_times(infile, relevant_venues, month = 1, length = 1000):

    longest     = {}

    for ind, line in enumerate(open(infile)):
        fields = line.strip().split('\t')
        venue  = fields[0]

        if venue in relevant_venues:



            times  = sorted([ round_unix_date(float(fff), month*30*24*60*60) for fff in fields[1:]])
            x, y = zip(* sorted([(k, v) for k, v in dict(Counter(times)).items()], key=lambda tup: tup[0]))       
            if len(x) > length:
                  longest[venue] = (x,y)

    return longest
    
    
def get_avg_counts(infile, month):

    times_count = {}
    for ind, line in enumerate(open(infile)):
        times  = [ round_unix_date(float(fff), month*30*24*60*60) for fff in line.strip().split('\t')[1:]]
        for time in times:
            if time not in times_count:
                times_count[time] = 1
            else:
                times_count[time] += 1   

    return times_count    


def completing_timeseries(timeseriesdict, all_times):
    
    timeseriesdict_out = {}
    
    for ind, (ven, (time, cnt)) in enumerate(timeseriesdict.items()):

        mint   =  min(time)
        time_x =  []
        cnt_x  =  []

        for ind, t in enumerate(all_times):
            if t > mint:
                if t in time:
                    time_x.append(t)
                    cnt_x.append(cnt[time.index(t)])
                else:
                    time_x.append(t)
                    cnt_x.append(0)

        x, y = zip(* sorted([(k, v) for k, v in zip(time_x, cnt_x)], key=lambda tup: tup[0]))       
    
        timeseriesdict_out[ven] = (x,y)      

    return timeseriesdict_out



def normalize_time_series(timeseries, timeseries_avg, total_avg):   
    
    venues_times_norm = {}

    for ind, (ven, (times, cnts)) in enumerate(timeseries.items()):
        
        tt = []
        cc = []
        
        for i in range(len(times)):
            normcnt  = float(cnts[i]) / timeseries_avg[times[i]] * total_avg
            tt.append(times[i])
            cc.append(normcnt)
            
        venues_times_norm[ven] = (tt, cc)


    venues_times_norm_norm = {}
    for ind, (ven, (times, cnt)) in enumerate(venues_times_norm.items()):
        maxcnt = np.mean(cnt)
        cnt    = [cc/maxcnt for cc in cnt] 
        venues_times_norm_norm[ven] = (times, cnt)

            
    return venues_times_norm_norm
        

def viz_timeseries(timeseries, ttitle):
    
    f, ax = plt.subplots(1, 1, figsize=(20, 6))

    for ind, (ven, times) in enumerate(timeseries.items()):

        if ind == 1000: break
        time_, cnt_ = zip(*[(t,c) for t,c in zip(*times) if c < 1000])
        ax.plot(time_, cnt_, '-', linewidth = 0.1225)


    ax.set_yscale('log')
    ax.set_xlabel('Time',fontsize = 16)
    ax.set_ylabel('Normalized monthly like count',fontsize = 16)
    ax.set_title(ttitle, fontsize = 20)

    plt.show()


def get_stretched_filtered_ts(longest_1m_12, limit_low, limit_up):
   
    lens = set()
    maxTT = max(max(c[0]) for c in longest_1m_12.values())
    longest_1m_12_norm_stretched = {}
    for ind, (venue, (time, count)) in enumerate(longest_1m_12_norm.items()):

        if len(time) > limit_low and len(time) < limit_up:
            lens.add(len(time))
            longest_1m_12_norm_stretched[venue] = (tuple(transform_ts(time, maxTT)), count) 


    print lens, len(longest_1m_12_norm_stretched)
    return longest_1m_12_norm_stretched










if __name__ == "__main__":


    city       = sys.argv[1]
    infile  = '../ProcessedData/' + city + '/venues_info/venues_time_series.dat'
    ofolder =  '../ProcessedData/' + city + '/timeseries'

    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    

    #relevant_venues = set([line.strip().split('\t')[0] for line in open('../ProcessedData/' + city + '/venues_info/venues_ward_full.dat') if 'venue' not in line])

    relevant_venues = [line.strip().split('\t')[0] for line in open(infile)]
    print 'relevant venues:  ', len(relevant_venues), '\n'




    longest_1m_12           = get_venues_times(infile, relevant_venues, month = 6, length = 1)
    avg_times_count_1m_12   = get_avg_counts(infile, month = 6)
    total_avg               = np.mean(avg_times_count_1m_12.values())
    all_times               = sorted(avg_times_count_1m_12.keys())
    longest_1m_12_norm      = normalize_time_series(longest_1m_12, avg_times_count_1m_12, total_avg)
    longest_1m_12_full      = completing_timeseries(longest_1m_12, all_times)
    longest_1m_12_full_norm = normalize_time_series(longest_1m_12_full, avg_times_count_1m_12, total_avg)





    print 'TOT: ', len(longest_1m_12_norm)
    junior = get_stretched_filtered_ts(longest_1m_12_norm, 1, 5)
    mid    = get_stretched_filtered_ts(longest_1m_12_norm, 4, 9)


    #viz_timeseries(junior, 'Juniors, #curves = ' + str(len(junior)))
    #viz_timeseries(mid,    'Mids,    #curves = ' + str(len(mid)))
    #viz_timeseries(senior, 'Seniors, #curves = ' + str(len(senior)))




    print 'junior  ' , len(junior)
    print 'mid     ' , len(mid)
    


    fout = open(ofolder + '/junior_timeseries_1_5.dat', 'w')
    for venue, timeseries in junior.items():
        fout.write( venue + '\t' + '\t'.join([str(fff) for fff in list(timeseries[1])])+ '\n')
    fout.close()


    fout = open(ofolder + '/mid_timeseries_4_9.dat', 'w')
    for venue, timeseries in mid.items():
        fout.write( venue + '\t' + '\t'.join([str(fff) for fff in list(timeseries[1])])+ '\n')
    fout.close()





    for ijk in [8,7,6,5,4]:

        print 'senior...', ijk

        senior = get_stretched_filtered_ts(longest_1m_12_norm, ijk, 13)

        fout = open(ofolder + '/senior_timeseries_' + str(ijk) + '_13.dat', 'w')
        for venue, timeseries in senior.items():
            fout.write( venue + '\t' + '\t'.join([str(fff) for fff in list(timeseries[1])])+ '\n')
        fout.close()


        print 'senior  ' , len(senior)








