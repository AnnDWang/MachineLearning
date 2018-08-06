# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 读取MNIST数据集，如果不存在会事先下载
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 把原始数据图片保存在MNIST_data/raw/文件夹下
# 如果没有这个文件夹，自动创建
save_dir='MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

# 保存前20张图片
for i in range(20):
    # 请注意 mnist.train.images[i,:]就表示第张图片（序号从0开始）
    image_array=mnist.train.images[i,:]
    # Tensorflow 中的MNIST图片是一个784维的相邻，把它还原成28x28的图像
    image_array=image_array.reshape(28,28)
    # 保存文件格式为:
    # mnist_train_0.jpg
    filename=save_dir+'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)