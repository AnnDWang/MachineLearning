# coding:utf-8
import tensorflow as tf

# 导入数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# x为训练图像的占位符，y_为训练图形标签的占位符
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

# 由于使用了卷积网络对图像进行分类，所以不能再使用784维的向量表示输入x
# 而是将其还原为28x28的图片形式,[-1,28,28,1]中-1表示形状的第一维根据x自动确定的
# 将单张图片从784维还原为28x28的矩阵图片
x_image=tf.reshape(x,[-1,28,28,1])

# 第一层卷积的代码如下：
# 可以返回一个给定形状的变量，并自动以截断正太分布初始化
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 返回一个给定形状的变量，初始化为0.1
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第一层卷积
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
# 真正进行卷积，卷积计算后调用ReLU作为激活函数
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
# 池化操作
h_pool1=max_pool_2x2(h_conv1)

# 第二次卷积计算
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

# 全连接层,输出为1024维的向量
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# 使用Dropput，keep_prob是一个占位符，训练时为0.5，测试时为1
keep_porb=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_porb)
# 在全连接层中加入了Dropout，是放置神经网络过拟合的一种手段
# 在每一步训练时，以一定的改了去掉网络中的某些连接，但这种去除不是永久性的，
# 只是在当前步骤中去除，并且每一步去除的连接都是随机选择的
# Dropout是0.5，也就是说训练时每一个连接都有50%的概率被去掉，在测试时保留所有连接

# 再加上一层全连接，把上一步得到的h_fc1_drop转换为10个类别的打分
# 把1024维向量转换为10维，对应10个类别
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

# 直接使用tensorflow提供的方法计算交叉熵损失

# 用tf.nn.softmax_cross_entropy_with_logits直接计算
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
# 同样定义train_step
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 创建session，对变量初始化
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练2000步
for i in range(2000):
    batch=mnist.train.next_batch(50)
    # 每100步报告一次在验证集上的准确率
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_porb:1.0})
        print('step %d, training accuracy %g' %(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_porb:0.5})

#训练结束后报告在测试集上的准确率
print('test accuray %g' %accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_porb:1.0}) )