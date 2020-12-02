#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:45:16 2020

@author: Waquar Shamsi
"""
######################################      PLOT THE DATASET      USING TSNE  (BOTH ORGINAL, TEST and TRAIN    ##################################
###########################################  MEMO   ########################################
  
import h5py
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

with h5py.File('MNIST_Subset.h5','r') as hf:
    print(list(hf.keys()))   # Found the key names, ['X' , 'Y']
    X = hf['X'][:]
    y = hf['Y'][:]
    #print(X,Y)
    
#print(X.shape,y.shape)      # (14251, 28, 28) and (14251,)

#SPLITTING THE DATA IN 80:20 RATIO WITH SEED 42
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#SAVE THE DATAFRAMES h5py
    #training set
hf = h5py.File('train_data_Q2.h5', 'w')
hf.create_dataset('X', data=X_train)
hf.create_dataset('y', data=y_train)
print("CREATED TRAINING DATASET")
hf.close()
    #test set
hf = h5py.File('test_data_Q2.h5', 'w')
hf.create_dataset('X', data=X_test)
hf.create_dataset('y', data=y_test)
print("CREATED TEST DATASET")
hf.close()


# DATA EXPLORATION:

print("DISTINCT CLASS LABELS:",set(y_train))
print("Count of samples from each class: ",Counter(y_train))
#BARPLOT
count_9 = Counter(y_train)[9]
count_7 = Counter(y_train)[7]
counts = []
counts.append(count_9)
counts.append(count_7)
fig = plt.figure(figsize = (10, 5)) 
plt.bar((7,9), counts, color ='blue', width = 0.4) 
plt.xlabel("Label") 
plt.ylabel("Number of Samples") 
plt.title("Class Distribution for Train Set") 
plt.show() 