# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd



'''  ---------------------------------------------------------  '''
'''                 the classifier                              '''
'''  ---------------------------------------------------------  '''

def classifiers(outfolder, city, X_train, y_train, X_test, pred_home, LIMIT):
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]    

  
    # preprocess dataset, split into training and test part
    #X = StandardScaler().fit_transform(X)
    

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clas  = clf.fit(X_train, y_train)
        pred_home['pred_home_' + name.replace(' ', '_')] = clas.predict(X_test)
 
    pred_home.to_csv(outfolder + '/user_homes/MLresults/' + city + '_ML_home_classification_RESULTS_' + str(LIMIT) + '.csv', sep='\t')
   
    

def classify_data(city, outfolder, LIMIT):

    print ('Do the MLhomepred classification...')

    ### training data budapest_ML_feature_has_home_TRAIN.csv
    df_train =  pd.read_csv(outfolder + '/user_homes/MLfeatures/' + city + '_ML_feature_has_home_TRAIN_' + str(LIMIT) + '.csv', index_col=0, sep = '\t')
    X_train  =  df_train.drop(columns = ['home', 'venue', 'user']).fillna(0)
    y_train  =  df_train['home']


    ### test data budapest_ML_feature_has_home_TEST
    df_test =  pd.read_csv(outfolder + '/user_homes/MLfeatures/' + city + '_ML_feature_has_home_TEST_' + str(LIMIT) + '.csv', index_col=0, sep = '\t')
    X_test  =  df_test.drop(columns = ['home', 'venue', 'user']).fillna(0)


    # get the output as home-venue-home-not-home pairs
    pred_home          = pd.DataFrame()
    pred_home['user']  = df_test['user']
    pred_home['venue'] = df_test['venue']


    classifiers(outfolder, city, X_train, y_train, X_test, pred_home, LIMIT) 




'''  ---------------------------------------------------------  '''
'''                 get results                                 '''
'''  ---------------------------------------------------------  '''

def conclude_class(city, outfolder, LIMIT):

    print ('Merge the MLhomepred results...')

    df    = pd.read_csv(outfolder + '/user_homes/MLresults/' + city + '_ML_home_classification_RESULTS_' + str(LIMIT) + '.csv', sep = '\t'  , index_col=0)
    f, ax = plt.subplots(2, 5, figsize=(20, 8))
    df_s  = df.groupby( ['user' ]).sum()
    df_c  = pd.DataFrame() 
    ind   = [(i,j) for i in range(2) for j in range(5)]


    for column, (k,l) in zip(df_s, ind):
        df_s[column].plot(kind = 'hist', ax = ax[k,l], logy = True, title = column)
        df_c[column.replace('pred_home_', '')] = df_s[column].value_counts()     


    f.savefig(outfolder   + '/figures/MLresults/' + city + '_ML_home_classification_distributions_' + str(LIMIT) + '.png')
    plt.close()
    df_c.to_csv(outfolder + '/user_homes/MLresults/' + city + '_ML_feature_has_home_RES_count_' + str(LIMIT) + '.csv' , sep = '\t')



    ''' GET THE PREDICTED HOME LOCATIONS FOR THE USERS'''

    df = pd.read_csv(outfolder + '/user_homes/MLresults/' + city + '_ML_home_classification_RESULTS_' + str(LIMIT) + '.csv', sep = '\t', index_col=0)

    data2 = {}
    for column in df:
        if 'pred_home' in column:

            ddf = df.groupby( ['user' ]).filter(lambda x: x[column].sum() == 1.)
            ddf = ddf.loc[df[column] == 1]
            ddf = ddf[['user', 'venue']]

            ddf.to_csv(outfolder + '/user_homes/MLhomes/' + city + '_predicted_homes_' + str(LIMIT) + '_' + column + '.csv' , sep = '\t', index = False)




    ''' FRACTION OF USERS WITH ML PREDICTED HOME LOCATIONS'''

    df = pd.read_csv(outfolder + '/user_homes/MLresults/' + city + '_ML_feature_has_home_RES_count_' + str(LIMIT) + '.csv' , sep = '\t', index_col=0).fillna(0)

    data = {}
    for column in df:
        data[column.replace('pred_home', '')] =round(100*df[column][1]/float(df[column].sum()), 2)
       
    f, ax  = plt.subplots(1, 1, figsize=(15, 5))
    names  = list(data.keys())
    values = list(data.values())

    ax.bar(range(len(data)),values,tick_label=names, label = 'Has 1 ML home location')
    ax.set_ylabel('% of users', fontsize= 20)


    f.savefig(outfolder   + '/figures/MLresults/' + city + '_ML_fraction_os_users_with_home_' + str(LIMIT) + '.png')
    ax.set_title('Fraction of users w exactly 1 predicted home location vs different methods', fontsize = 22)
    plt.close()











