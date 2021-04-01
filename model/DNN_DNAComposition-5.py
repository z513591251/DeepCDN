from __future__ import print_function
import numpy as np
import pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras import optimizers
from keras import backend as K
from Filereader import readFile
from DNAFeature import DNAComposition

result = []
for i in range(1,11):
     round = 'Round{num}'.format(num=i)
     print(round)
     print('Loading data.......................................')
     trainfile = 'train{num}.txt'.format(num=i)
     testfile = 'test{num}.txt'.format(num=i)
     (train_seq,train_value)= readFile(trainfile)
     print('Setting the parameters.......................................')
     length = int(len(train_seq[0]))
     batch_size = 40
     epochs = 150

     hidden_dims_1 = 682
     hidden_dims_2 = 341
     hidden_dims_3 = 170
 
     print('Building model...............................................')
     train_model = 'model{num}'.format(num=i)
     train_model = Sequential()
     train_model.add(Dense(hidden_dims_1, activation = 'relu'))
     train_model.add(Dense(hidden_dims_2, activation = 'relu'))
     train_model.add(Dense(hidden_dims_3, activation = 'relu'))
     train_model.add(Dense(1,activation='linear'))
     train_model.compile(optimizer = 'sgd',loss='mse')

     print('Generating features..........................................')
     x_train_1= DNAComposition(train_seq,1)
     x_train_2= DNAComposition(train_seq,2)
     x_train_3= DNAComposition(train_seq,3)
     x_train_4= DNAComposition(train_seq,4)
     x_train_5= DNAComposition(train_seq,5)
     x_train= np.hstack((x_train_1,x_train_2,x_train_3,x_train_4,x_train_5))
     y_train = np.array(train_value)
     print('Training model..............................................')
     train_model.fit(x_train, y_train,batch_size = batch_size,epochs = epochs)
     print('Predicting..............................................')
     (test_seq,test_value)= readFile(testfile)
     x_test_1= DNAComposition(test_seq,1)
     x_test_2= DNAComposition(test_seq,2)
     x_test_3= DNAComposition(test_seq,3)
     x_test_4= DNAComposition(test_seq,4)
     x_test_5= DNAComposition(test_seq,5)
     x_test= np.hstack((x_test_1,x_test_2,x_test_3,x_test_4,x_test_5))
     y_test = np.array(test_value)
     y_test = y_test.reshape(y_test.shape[0],1) 
     predicted = train_model.predict(x_test)
     combine = np.hstack((y_test,predicted))
     for num in combine:
         result.append(num)


print('Summarizing the result............................................')
result = np.array(result).reshape(len(result),2)
np.savetxt('1.csv',result,fmt="%s",delimiter=",")


print('Showing the result................................................')
print(np.corrcoef(result[:,0],result[:,1])[0,1])
plt.scatter(result[:,0],result[:,1])
plt.show()
     

     
     