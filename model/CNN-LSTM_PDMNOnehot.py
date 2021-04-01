from __future__ import print_function
import numpy as np
import pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras import optimizers
from keras import backend as K
from Filereader import readFile
from DNAFeature import PDMNOnehot


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
     pool_size = 2
     lstm_output_size = 64
     filters = 100
     batch_size = 40
     epochs = 150

     if length % 2 == 0:
        kernel_size = 3
     else:
        kernel_size = 2

     print('Building model...............................................')
     train_model = 'model{num}'.format(num=i)
     train_model = Sequential()
     train_model.add(Reshape((length,4),input_shape = (length*4,)))
     train_model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
     train_model.add(MaxPooling1D(pool_size = pool_size))
     train_model.add(LSTM(lstm_output_size))
     train_model.add(Dense(1,activation='linear'))
     train_model.compile(optimizer = 'sgd',loss='mse')

     
     print('Generating features..........................................')
     x_train = PDMNOnehot(train_seq)
     y_train = np.array(train_value)
     print('Training model..............................................')
     train_model.fit(x_train, y_train,batch_size = batch_size,epochs = epochs)
     print('Predicting..............................................')
     (test_seq,test_value)= readFile(testfile)
     x_test = PDMNOnehot(test_seq)
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


     
        
