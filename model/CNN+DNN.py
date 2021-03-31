from __future__ import print_function
import numpy as np
import pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers
from keras.models import Model
from keras import backend as K
from Filereader import readFile
from DNAFeature import PDMNOnehot
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
     pool_size = 2
     filters = 100
     batch_size = 40
     epochs = 150

     if length % 2 == 0:
        kernel_size = 3
     else:
        kernel_size = 2

     hidden_dims_1 = int((((length-kernel_size+1)/2)*filters)/2)
     hidden_dims_2 = int(hidden_dims_1/2)
     hidden_dims_3 = int(hidden_dims_2/2)
     hidden_dims_4 = 682
     hidden_dims_5 = 341
     hidden_dims_6 = 170


     print('Building model...............................................')
     input_1 = Input(shape = (length*4,))
     x_1 = Reshape((length,4),input_shape=(length*4,))(input_1)
     x_2 = Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1)(x_1)
     x_3 = MaxPooling1D(pool_size = pool_size)(x_2)
     x_4 = Flatten()(x_3)
     x_5 = Dense(hidden_dims_1, activation = 'relu')(x_4)
     x_6 = Dense(hidden_dims_2, activation = 'relu')(x_5)
     output_1 = Dense(hidden_dims_3, activation = 'relu')(x_6)
     
     input_2 = Input(shape = (1364,))
     x_7 =  Dense(hidden_dims_4, activation = 'relu')(input_2)
     x_8 = Dense(hidden_dims_5, activation = 'relu')(x_7)
     output_2 = Dense(hidden_dims_6, activation = 'relu')(x_8)
     
     model = Concatenate()([output_1,output_2])  
     out = Dense(1,activation='linear')(model)
     new_model = Model([input_1,input_2],out)
     new_model.compile(optimizer = 'sgd',loss='mse')
     

     
     print('Generating features..........................................')
     x_train_1 = PDMNOnehot(train_seq)
     y_train = np.array(train_value)
     x_train_2 = DNAComposition(train_seq,1)
     x_train_3 = DNAComposition(train_seq,2)
     x_train_4 = DNAComposition(train_seq,3)
     x_train_5 = DNAComposition(train_seq,4)
     x_train_6 = DNAComposition(train_seq,5)
     x_train_7 = np.hstack((x_train_2,x_train_3,x_train_4,x_train_5,x_train_6))
     

     print('Training model..............................................')
     new_model.fit([x_train_1,x_train_7],y_train,batch_size = batch_size,epochs = epochs)
     print('Predicting..............................................')
     (test_seq,test_value)= readFile(testfile)
     x_test_1 = PDMNOnehot(test_seq)
     y_test = np.array(test_value)
     x_test_2 = DNAComposition(test_seq,1)
     x_test_3 = DNAComposition(test_seq,2)
     x_test_4 = DNAComposition(test_seq,3)
     x_test_5 = DNAComposition(test_seq,4)
     x_test_6 = DNAComposition(test_seq,5)
     x_test_7 = np.hstack((x_test_2,x_test_3,x_test_4,x_test_5,x_test_6))
     y_test = y_test.reshape(y_test.shape[0],1) 
     predicted = new_model.predict([x_test_1,x_test_7])
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


     
        
