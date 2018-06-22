import os
from ParseJsons import check_box


def copy_filtered(city, outfolder, bbox):


    print ('Drop locations of of the bounding box...')

    ''' filter out the centroid-based screwed up home locations '''

    folders_i = outfolder + '/user_homes/centroids/'
    folders_o = outfolder + '/user_homes/centroids_filtered/'

    if not os.path.exists(folders_o):
        os.makedirs(folders_o)

    for ind, fn in enumerate(os.listdir(folders_i)):
        
        fout = open(folders_o + fn.replace('.dat', '') + '_filtered.dat', 'w') 

        for line  in open(folders_i + fn):
            user, lng, lat = line.strip().split('\t')
           
            if check_box(bbox, city, float(lat), float(lng)):
                fout.write(line)


    fout.close()


   

    ''' get the coordinates of the ML home locations and drop those out of the city '''

    venues_coordinates = {}
    
    for ind, line in enumerate(open(outfolder + '/user_info/bristol_user_venues_full_locals_filtered.dat')):
        
        venues = line.strip().split('\t')[1:]

        for v in venues:
            vid, lng, lat, cat = v.split(',')
            
            if vid not in venues_coordinates:
                venues_coordinates[vid] = (lng, lat)



    ML_folder_i = outfolder + '/user_homes/MLhomes/'
    ML_folder_o = outfolder + '/user_homes/MLhomes_filtered/'

    if not os.path.exists(ML_folder_o):
        os.makedirs(ML_folder_o)

    
    for ind, fn in enumerate( os.listdir(ML_folder_i) ):     
        
        fout = open(ML_folder_o + fn.replace('.csv', '') + '_filtered.dat', 'w') 

        for line in open(ML_folder_i + fn):

            if 'user' not in line:
                user, venue = line.strip().split('\t')
                lng, lat = venues_coordinates[venue]
             
                if check_box(bbox, city, float(lat), float(lng)):

                    fout.write(user + '\t' + lng + '\t' + lat + '\n')


        fout.close()










   
