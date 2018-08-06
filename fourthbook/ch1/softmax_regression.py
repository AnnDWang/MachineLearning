# 导入tensorflow
import tensorflow as tf
# 导入数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

#创建x,x是一个占位符，表示待识别的图片
x=tf.placeholder(tf.float32,[None,784])

# W是softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在Tensorflow中，变量的参数用tf.Varible表示
W=tf.Variable(tf.zeros([784,10]))

# b是有一个softmax模型的参数，一般叫做偏置项
b=tf.Variable(tf.zeros([10]))

# y表示模型的输出
y=tf.nn.softmax(tf.matmul(x,W)+b)
# y_是实际的图形，同样以占位符表示
y_=tf.placeholder(tf.float32,[None,10])

# 至此，得到了两个重要的Tensor：y和y_
# y是模型的输出，y_是实际的图像标签，y_是独热表示的
# 根据y和y_构造损失

# 根据y和Y_构造交叉熵损失
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))

# 下一步是优化损失，让损失减小。这里使用梯度下降法

# 有了损失，就可以用梯度下降法针对模型的参数（W和b）进行优化
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# tensorflow 默认会对所有的变量计算梯度，在这里之定义了两个变量W和b，因此程序将会使用梯度下降法对W、b计算梯度并更新它们的值

# 创建一个session，只有在session中才能运行优化步骤train_step
sess=tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # btach_xs是形状在（100，784）的图像数据，batch_ys是形如（100，10）的实际标签
    # batch_xs,batch_ys对应着两个占位符x,y_
    batch_xs,batch_ys=mnist.train.next_batch(100)
    # 在session中运行train_step，运行时要传入占位符的值
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# 正确的预测结果
# y的形状是(N,10),y_是（N,10）
# 其中N为输入模型的样本数
# tf.argmax(y,1)功能是取出数组中最大值的下标
# 可以用来将独热表示以及模型输出转换为数字标签。
# 假设y_为[[1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,0]]
# 则tf.argmax(y_,1)为[0,2,9,0]，若tf.argmax(y,1)为【0，0，0，0】
# 则 correct_prediction为[True,False,False,True]
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 计算预测准确率，都是Tensor
# tf.cast(correct_prediction,tf.float32)将比较值转为float32型的变量
# True被转换为1，False被转换为0
# tf.reduce_mean可以计算数组中所有元素的平均值，相当于得到了模型预测准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 在session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的准确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

