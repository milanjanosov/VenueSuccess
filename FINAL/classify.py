import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score



'''
def xgb_model_params(train_data, train_label, test_data, test_label, max_depth_, learning_rate_, subsample_):
     
    model2       = xgb.XGBClassifier(n_estimators=1000, max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
    train_model2 = model2.fit(train_data, train_label)
    pred2        = train_model2.predict(X_test)
    

    
    return accuracy_score(y_test, pred2) * 100
'''       
 


def xgb_model_params(X, y, max_depth_, learning_rate_, subsample_):
     
        
    train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)            
        
    model2       = xgb.XGBClassifier(n_estimators=1000, max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
    train_model2 = model2.fit(train_data, train_label)
    pred2        = train_model2.predict(test_data)
    
    return accuracy_score(test_label, pred2) * 100


features = pd.read_csv('FINAL_FEATURES_UNCORR_BALANCED_160.csv', sep = '\t', index_col = 0)   



X = features.drop(columns = ['LABEL_category'])
y = np.asarray(features.LABEL_category)



fout = open('gridsearch.dat', 'w')

#X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.33, random_state=42)

results = []


# (5, 0.55, 0.40000000000000013, 31.413612565445025)


depths = [4,5,6]
ress1  = list(np.arange(0.0, 0.5, 0.05))  #list(np.arange(0.5, 1.0, 0.01))
ress2  = list(np.arange(0.5, 1.0, 0.05))  #list(np.arange(0.25, 0.75, 0.01))
nnn    = len(depths) * len(ress1) * len(ress2)
ijk    = 1

for max_depth_ in depths:
    
    for learning_rate_ in ress1:
        
        for subsample_ in ress2:


            print ijk, learning_rate_, subsample_,  '/', nnn
            ijk += 1
        
            a = xgb_model_params(X, y, max_depth_, learning_rate_, subsample_)

            results.append((max_depth_, learning_rate_, subsample_, a))
    
            fout.write(str(max_depth_) + '\t' + str(learning_rate_) + '\t' + str(subsample_) + '\t' + str(a) + '\n')


fout.close()

results_s = sorted(results, key=lambda tup: tup[3], reverse = True)

print results_s[0]






