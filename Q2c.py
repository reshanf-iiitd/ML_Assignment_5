#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:26:57 2020

@author: Waquar Shamsi
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:26:57 2020

@author: Waquar Shamsi
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pickle
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import os

alphas = (1,0.0001,0.000000001)
for alpha in alphas:
    if os.path.exists('pca_MLPClassifier'+str(alpha)): # IF THE MODEL IS ALREADY TRAINED AND AVAILABLE
        clf = pickle.load(open('pca_MLPClassifier'+str(alpha), 'rb'))
        print("TRAINED MODEL EXISTS AND IS LOADED")
    else:       #IF MODEL NOT ALREADY EXISTS THEN CREATE ITS
        print("TRAINED MODEL IS NOT AVAILABLE, TRAINING NOW...")
    
        with h5py.File('train_data_Q2.h5','r') as train_file:   #LOAD THE TRAIN SET
            X_train=train_file['X'][:]
            y_train=train_file['y'][:]
    
        X_train = X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[1]])
    
        pca = PCA(n_components=2)
        pca.fit(X_train)
        X_pca = pca.transform(X_train)
    #    
    #    #ONE HOT ENCODER
    #    ohe = OneHotEncoder()
    #    y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
    #    
        clf = MLPClassifier(max_iter=1000,alpha=alpha,hidden_layer_sizes=(100, 50,50),
                            learning_rate_init=0.001,activation='logistic')
      
        #FIT THE MODEL
        clf.fit(X_pca, y_train)
      
        #SAVE THE MODEL
        pickle.dump(clf, open('pca_MLPClassifier'+str(alpha), 'wb'))

    
    with h5py.File('test_data_Q2.h5','r') as train_file:
        X_test=train_file['X'][:]
        y_test=train_file['y'][:]
    
    
    X_test = X_test.reshape([X_test.shape[0],X_test.shape[1]*X_test.shape[1]])
    
    pca = PCA(n_components=2)
    pca.fit(X_test)
    X_pca = pca.transform(X_test)
      
#    #ONE HOT ENCODER
#    ohe = OneHotEncoder()
#    y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
#    
    #PREDICT
    pred = clf.predict(X_pca)
    
    #FEATURE REDUCTION TO 2 FEATURES USING PCA - TO PLOT IN 2D
    
    #print(X_pca.shape)
    #print(X_test.shape)
    plt.figure(figsize=(20,10))
    ax = plt.gca()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=pred, s=25, edgecolor='k')
    ax.set_title('Decision Boundary with alpha='+str(alpha))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    
    x=X_pca[:, 0]
    y=X_pca[:, 1]
    h=.02
    
        
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    ax.contourf(XX, YY, Z, mode='filled', alpha=0.4)    
    ax.set_xlim((np.min(X_pca[:, 0]), np.max(X_pca[:, 0])))
    ax.set_ylim((np.min(X_pca[:, 1]), np.max(X_pca[:, 0])))
