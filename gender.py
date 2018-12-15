import os
import json
import sys

city = sys.argv[1]

fout = open('../ProcessedData/london/user_info/users_gender.dat', 'w')

for ind, line in enumerate(open('../Data/fsqdb/london/'+city+'_users.json')):

    #if ind == 2 : break

    data = json.loads(line)



    fout.write( data['id'] + '\t' + data['gender'] + '\n')

fout.close()

