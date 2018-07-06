import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score



class LinearRegression(linear_model.LinearRegression):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self




def get_features_targets(ok_feats):

    ### success measures 
    city        = 'london'
    outfolder   = '../ProcessedData/' + city + '/'
    successdata = outfolder + 'venues_info/' + city + '_venues_success_measures.csv'
    success     = pd.read_csv(successdata, sep = '\t', index_col = 0)


    ### ward level features 
    outfolder    = '../ProcessedData/' + city + '/'
    ward_stats_f = outfolder + 'venues_info/venues_ward_full.dat'
    ward_stats   = pd.read_csv(ward_stats_f, sep = '\t', index_col = 0).drop(['ward'], axis=1)


    ### category features 
    ward_cats_f  = outfolder + 'venues_info/' + city + '_WARD_category_stats.csv'
    ward_cats    = pd.read_csv(ward_cats_f, sep = '\t', index_col = 0).drop(['ward'], axis=1)


    ### network features 
    network_meas_f  = outfolder + '/networks/' + city + '_COMBINED_networkmeasures.csv'
    network_meas    = pd.read_csv(network_meas_f, sep = ',', index_col = 0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


    # merge and filter full feature set
    temp_features     = network_meas.join(ward_cats)
    venue_features    = temp_features.join(ward_stats)
    filtered_features = pd.DataFrame()

    for feat in ok_feats:
        filtered_features[feat] = venue_features[feat]
        
    for feat in ok_feats:
        filtered_features = filtered_features[filtered_features[feat] != 0.0]    


    # scale features and targets
    X_head = filtered_features.keys()
    X      = filtered_features.values
    X      = StandardScaler().fit_transform(X)
    X      = preprocessing.quantile_transform(X, output_distribution = 'normal')


    success_t = success.reset_index()
    success_t = success_t[success_t['index'].isin(list(filtered_features.index)) ]

    y = np.asarray(success_t['checkinsCount'])
    y = StandardScaler().fit_transform(y.reshape(-1, 1) )
    y = preprocessing.quantile_transform(y, output_distribution = 'normal')

    yy = np.asarray(success_t['tipCount'])
    yy = StandardScaler().fit_transform(y.reshape(-1, 1) )
    yy = preprocessing.quantile_transform(y, output_distribution = 'normal')


    return X, y, X_head, yy



def do_linear_regression(X, y, X_head):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Create linear regression object
    linear_regressor = LinearRegression()

    # Train the model using the training sets
    res = linear_regressor.fit(X_train, y_train)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, linear_regressor.predict(X_test)))
    # Explained variance score: 1 is perfect prediction
    print('R^2 (training): %f' % r2_score(y_train, linear_regressor.predict(X_train)))
    print('R^2 (testing):  %f' % r2_score(y_test, linear_regressor.predict(X_test))), '\n'

    for i in range(len(linear_regressor.coef_[0])):
        print X_head[i], '\t\tcoeff = ', linear_regressor.coef_[0][i], '\t p = ',linear_regressor.p[0][i]

    return r2_score(y_test, linear_regressor.predict(X_test))







def do_stuff(ok_feats):

    X, y, X_head, yy = get_features_targets(ok_feats)
    R2    = do_linear_regression(X, y, X_head)
    R2ref = do_linear_regression(yy, y, X_head)

    return R2, R2ref




all_features = ['closeness_entropy',  'social_stretch_entropy',  'egosize_entropy',  'geo_size_of_ego_entropy',  'pagerank_geo_entropy',  'closeness_geo_entropy',  'lng',  'lat',  'constraint_std',  'constraint_geo_entropy',  'constraint_entropy',  'degree_entropy',  'constraint_geo_std',  'geo_stdev_of_ego_entropy',  'betweenness_geo_entropy',  'strength_geo_entropy',  'betweenness_entropy',  'pagerank_entropy',  'geo_stdev_of_ego_std',  'social_stretch_std',  'eigenvector_geo_entropy',  'eigenvector_entropy',  'clustering_entropy',  'closeness_avg',  'pagerank_geo_std',  'clustering_geo_entropy',  'geo_size_of_ego_std',  'clustering_geo_std',  'closeness_geo_avg',  'social_stretch_avg',  'constraint_geo_avg',  'egosize_std',  'eigenvector_avg',  'pagerank_std',  'constraint_avg',  'pagerank_avg',  'eigenvector_geo_avg',  'geo_stdev_of_ego_avg',  'degree_std',  'pagerank_geo_avg',  'strength_geo_avg',  'clustering_std',  'clustering_avg',  'betweenness_avg',  'betweenness_geo_avg',  'clustering_geo_avg',  'degree_avg',  'egosize_avg',  'betweenness_geo_std',  'travel',  'strength_geo_std',  'geo_size_of_ego_avg',  'lng1',  'shops',  'lng0',  'education',  'length',  'arts_entertainment',  'lat0',  'Venues_num',  'lat1',  'building',  'nightlife',  'area',  'closeness_std',  'eigenvector_geo_std',  'food',  'Entropy',  'parks_outdoors',  'eigenvector_std',  'na',  'event',  'closeness_geo_std',  'betweenness_std']    


top8_features   = ['closeness_entropy', 'social_stretch_entropy', 'egosize_entropy', 'geo_size_of_ego_entropy', 'pagerank_geo_entropy', 'closeness_geo_entropy', 'lng','lat']
top8_network    = ['closeness_entropy', 'social_stretch_entropy', 'egosize_entropy', 'geo_size_of_ego_entropy', 'pagerank_geo_entropy', 'closeness_geo_entropy', 'constraint_std', 'constraint_geo_entropy']
top8_nonnetwork = ['lng', 'lat', 'travel', 'lng1', 'shops', 'lng0', 'education', 'length']




R_all, R2ref = do_stuff(all_features)



R2_nw,  R2_nw_ref  = do_stuff(top8_network)
R2_all, R2_all_ref = do_stuff(all_features)
R2_top, R2_top_ref = do_stuff(top8_features)
R2_nnw, R2_nnw_ref = do_stuff(top8_nonnetwork)


print '\n\n R^2 values:\t'

print R2_nw,  R2_nw_ref
print R2_all, R2_all_ref
print R2_top, R2_top_ref
print R2_nnw, R2_nnw_ref 




