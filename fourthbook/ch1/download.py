# coding:utf-8
# 这是从tensorflow.examples.tutorials.mnist引入模块
# 这是Tensorflow为了教学MNIST而提供的程序
from tensorflow.examples.tutorials.mnist import input_data
# 从MNIST_data从读取MNIST数据，折腾语句在数据不存在时，会自动执行下载
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 查看训练数据的大小
print(mnist.train.images.shape )#(55000,784)
print(mnist.train.labels.shape)#(55000,10)

# 查看验证数据集的大小
print(mnist.validation.images.shape) #(5000,784)
print(mnist.validation.labels.shape)#(5000,10)

# 查看测试数据集大小
print(mnist.test.images.shape)# (10000,784)
print(mnist.test.labels.shape)# (10000,10)