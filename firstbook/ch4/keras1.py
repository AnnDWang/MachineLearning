from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size=128 # 梯度下降一个批的数据量
nb_classes=10# 类别
nb_epoch=10# 梯度下降epoch循环训练次数，每次循环包括全部训练样本
img_size=28*28
# 加载数据，已执行suffle-split 训练-测试集随机分割
(X_train,y_train),(X_test,y_test)=mnist.load_data()
# 以tensorflow，归一化输入数据，生成图片向量
X_train=X_train.reshape(y_train.shape[0],img_size).astype('float32')/255
X_test=X_test.reshape(y_test.shape[0],img_size).astype('float32')/255

print (X_train.shape,X_test.shape)

# one-hot 编码标签，将【3,2，。。。】变成[[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],...]
Y_train=np_utils.to_categorical(y_train,nb_classes)
Y_test=np_utils.to_categorical(y_test,nb_classes)

# 创建模型，逻辑分类相当于一层全连接的神经网络 Desnse 是Keras中定义的DNN模型
model=Sequential([Dense(10,input_shape=(img_size,),activation='softmax'),])