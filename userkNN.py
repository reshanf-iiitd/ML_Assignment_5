###############################################################################################
#####                                   Q1-2                                              #####
###############################################################################################
import numpy as np

class userkNN:
  K=3
  def __init__(self,k):
    self.K=k

  def knn_predictions(self,xTrain,yTrain,xTest):
      indices, distances = self.knn_distances(xTrain,xTest,self.K)
      yTrain = yTrain.flatten()
      rows, columns = indices.shape
      predictions = list()
      for j in range(columns):
          temp = list()
          for i in range(rows):
              cell = indices[i][j]
              temp.append(yTrain[cell])
          predictions.append(max(temp,key=temp.count)) #this is the key function, brings the mode value
      predictions=np.array(predictions)
      return predictions
    
  def knn_distances(self,xTrain,xTest,k):
      distances = -2 * xTrain@xTest.T + np.sum(xTest**2,axis=1) + np.sum(xTrain**2,axis=1)[:, np.newaxis]
      distances[distances < 0] = 0
      distances = distances**.5
      indices = np.argsort(distances, 0) #get indices of sorted items
      distances = np.sort(distances,0) #distances sorted in axis 0
      #returning the top-k closest distances.
      return indices[0:k,:], distances[0:k,:]
    
