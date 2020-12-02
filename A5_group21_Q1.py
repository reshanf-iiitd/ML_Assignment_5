
import pandas as pd
from sklearn.manifold import TSNE 
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from userkNN import userkNN


################### HELPER FUNCTIONS  ###################################
def test_30(X,y):
  m, n = X.shape
  X_test = X[:int(np.floor(m*0.3))]
  X_train = X[int(np.floor(m*0.3)):]

  y_test = y[:int(np.floor(m*0.3))]
  y_train = y[int(np.floor(m*0.3)):]

  return X_test,y_test,X_train,y_train

   
# Calculate accuracy percentage
def knn_accuracy(yTest,predictions):
  x=yTest.flatten()==predictions.flatten()
  grade=np.mean(x)
  return np.round(grade*100,2)
##########################################################################

def Q1_1():
	########################### TESTING to create the Class  ###########################################################

	data=pd.read_csv('sat.trn',sep = ' ')
	# print(mat1)
	y=data['3']
	# print(data)
	X=data.drop(labels='3',axis=1)
	# print(data)
	# X_test,y_test,X_train,y_train=test_20(s2,l2)
	model = TSNE(n_components = 2, random_state = 0,perplexity=30,n_iter=1000) 
	tsne_data = model.fit_transform(X)
	tsne_data = np.vstack((tsne_data.T, y)).T 
	tsne_df = pd.DataFrame(data = tsne_data, 
	      columns =("Dim_1", "Dim_2", "label"))

	sns.FacetGrid(tsne_df, hue ="label", size = 8).map( 
	        plt.scatter, 'Dim_1', 'Dim_2')
	plt.legend(loc=4)
	plt.title("Scattered Plot by reducing dimension using t-SNE",fontsize=20,color="w")
	plt.tight_layout()
	# plt.savefig('fig_1_3.jpg')
	plt.show()

	##########################################  3D      #################

	# model3 = TSNE(n_components = 3, random_state = 0,perplexity=30,n_iter=1000) 
	# tsne_data3 = model3.fit_transform(X)
	# tsne_data3 = np.vstack((tsne_data3.T, y)).T 
	# tsne_df3 = pd.DataFrame(data = tsne_data3, 
	#       columns =("Dim_1", "Dim_2","Dim_3", "label"))

	# fig = px.scatter_3d(tsne_df3,x="Dim_1",y="Dim_2",z="Dim_3",
 #                  color='label',title="Scattered Plot by reducing to 3dimension using t-SNE",width=1000, height=800)
	# fig.show()




def Q1_2():
	data_train=pd.read_csv('sat.trn',sep = ' ')
	data_test=pd.read_csv('sat.tst',sep = ' ')

	data_test.isnull().values.any()
	ytrain=data_train['3']
	# print(data)
	Xtrain=data_train.drop(labels='3',axis=1)

	ytest=data_test['3']
	# print(data)
	Xtest=data_test.drop(labels='3',axis=1)

	ytrain=np.array(ytrain)
	ytest=np.array(ytest)
	Xtrain=np.array(Xtrain)
	Xtest=np.array(Xtest)

	knnnn = userkNN(10)

	predictions = knnnn.knn_predictions(Xtrain, ytrain, Xtest)

	print('Accuracy:',knn_accuracy(predictions,ytest),'%')

	######################### APPLYING GRID SEARCH TO FIND THE OPTIMAL K


	error =[]

	for x in range(1,101):
	  # print(x)
	  knnnn = userkNN(k=x)
	  predictions1 = knnnn.knn_predictions(Xtrain, ytrain, Xtest)
	  temp = knn_accuracy(predictions1,ytest)
	  error.append(100-temp)
	  # print(temp)

	val_k=[x for x in range(1,101)]
	df=pd.DataFrame({'Number of Neighbours(k)':val_k ,
	                 "Error":error,
	                
	})
	print(tabulate(df, headers='keys', tablefmt='psql',showindex='never'))

	############################# Finding optimal K 
	print("Maximum Accuracy is ",(100-min(error) ))
	print("Optimal Value of K is ",error.index(min(error)) + 1 )

	plt.figure(figsize=(11,7))
	plt.plot(val_k,error,'g', color='red')
	plt.ylabel('Error % ')
	plt.title(" Error vs Number of neighbours graph (k)")
	plt.xlabel('Number of Neighbors (k)')
	plt.tight_layout()
	plt.yticks(weight='bold')

	plt.xticks([x for x in range(1,101,3)],weight='bold')
	plt.show()


def Q1_3():
		##################################### Q1-3 ###########################
	######### OPTIMAL K=4
	#################3 USER KNN ########################## ACCURACY

	data_train=pd.read_csv('sat.trn',sep = ' ')
	data_test=pd.read_csv('sat.tst',sep = ' ')

	data_test.isnull().values.any()
	ytrain=data_train['3']
	# print(data)
	Xtrain=data_train.drop(labels='3',axis=1)

	ytest=data_test['3']
	# print(data)
	Xtest=data_test.drop(labels='3',axis=1)

	ytrain=np.array(ytrain)
	ytest=np.array(ytest)
	Xtrain=np.array(Xtrain)
	Xtest=np.array(Xtest)

		#########################  USER DEFINED
	knnnn = userkNN(4)
	predictions = knnnn.knn_predictions(Xtrain, ytrain, Xtest)
	print('Validation Accuracy user defined KNN:',knn_accuracy(predictions,ytest),'%')

	knnnn = userkNN(4)
	predictions = knnnn.knn_predictions(Xtrain, ytrain, Xtrain)
	print('Training Accuracy user defined KNN:',knn_accuracy(predictions,ytrain),'%')

	#########################  Sklearn #########################
	neigh = KNeighborsClassifier(n_neighbors = 4).fit(Xtrain,ytrain)      ###########   algorithm is auto hence it will select based value passed to fit()
	predd=neigh.predict(Xtest)
	print('Validation Accuracy using sklearn:',knn_accuracy(predd,ytest),'%')

	neigh = KNeighborsClassifier(n_neighbors = 4).fit(Xtrain,ytrain)
	predd=neigh.predict(Xtrain)
	print('Validation Accuracy using sklearn:',knn_accuracy(predd,ytrain),'%')




    
    
    #Train Model and Predict
    #neigh = KNeighborsClassifier(n_neighbors = n).fit(xTrain,yTrain)
    #yhat=neigh.predict(xTest)









if __name__ == "__main__":
    print("PhD19006")
    # Q1_1()
    Q1_2()
    Q1_3()