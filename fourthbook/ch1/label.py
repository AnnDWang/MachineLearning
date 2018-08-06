# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 读取MNIST数据集，如果不存在会先下载
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 看前20张图片的label
for i in range(20):
    # 得到独热表示（0，1，0，0，0，0，0，0，0，0）
    # 独热表示指一位有效编码，用N维向量来表示N个类别，每个类别占据独立的一位
    # 任何时候独热表示只有一位是1，其他都为0
    one_hot_label=mnist.train.labels[i:]
    # 通过np.argmax可以直接获得原始label
    label=np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label : %d' %(i,label))