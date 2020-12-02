#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 01:30:31 2020

@author: Waquar Shamsi
"""

################## PLOT LOSS ATTRIBUTE OF MLP CLASSIFIERE for each iteration in gradient ######################### 
#######################    PERFORM GRID SEARCH GRIDSEARCHCV ###################################
############################### MEMO ##############################
from sklearn.neural_network import MLPClassifier
import h5py
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


with h5py.File('train_data_Q2.h5','r') as train_file:   #LOAD THE TRAIN SET
        X_train=train_file['X'][:]
        y_train=train_file['y'][:]
    
      
#print(X.shape,y.shape) # (11400, 28, 28) and (11400,)

# print(np.unique(y_train)) #[7 9] only two unique classes
X_train = X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[1]])
#print(X_train.shape)

#FEATURE SCALING
fs = StandardScaler()
X_train = fs.fit_transform(X_train)
   
# ONE HOT ENCODING
ohe = OneHotEncoder() 
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()


alphas = [0.0001,0.001,0.1,10]
if os.path.exists('MLPClassifier'): # IF THE MODEL IS ALREADY TRAINED AND AVAILABLE
    clf = pickle.load(open('MLPClassifier', 'rb'))
    print("TRAINED MODEL EXISTS AND IS LOADED")
else:       #IF MODEL NOT ALREADY EXISTS THEN CREATE ITS
    print("TRAINED MODEL IS NOT AVAILABLE, TRAINING NOW...")
    
    
    # GRID SEARCH
    
    parameters = {'max_iter': [1000], 'alpha': alphas,
                  'hidden_layer_sizes':[(100, 50,50)],'learning_rate_init':[0.001],'activation':['logistic']}
    clf = GridSearchCV(MLPClassifier(), parameters,cv=2)
    
    #FIT THE MODEL
    clf.fit(X_train, y_train)
    
    #SAVE THE MODEL
    pickle.dump(clf, open('MLPClassifier', 'wb'))
 

# LOAD TEST DATA
with h5py.File('test_data_Q2.h5','r') as train_file:
    X_test=train_file['X'][:]
    y_test=train_file['y'][:]

X_test = X_test.reshape([X_test.shape[0],X_test.shape[1]*X_test.shape[1]])

#FEATURE SCALING
X_test = fs.fit_transform(X_test)

#ONE HOT ENCODER
ohe = OneHotEncoder()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()

#PREDICT
pred = clf.predict_proba(X_test)

#PRINT METRICS 
metrics = [
        ["Training set score",clf.score(X_train, y_train)],
        ["Test set score",clf.score(X_test, y_test)],
        ["Training Loss",clf.best_estimator_.loss_],
        ["Testing Loss:",log_loss(y_test,pred)]
        ]
print(tabulate(metrics, ("Criteria","Value"),tablefmt="grid"))

plt.title("Accuracy vs Alpha")
plt.plot(alphas,clf.cv_results_['mean_test_score'],label='Accuracy')
plt.xlabel("ALPHA VALUES")
plt.ylabel("TEST ACCURACIES")
plt.legend()
plt.show()