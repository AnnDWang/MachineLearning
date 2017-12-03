#1.4小节 逻辑分类2：线性分类模型
import numpy as np

#函数y=1*x^2+0*x+0
y=np.poly1d([1,0,0])
y(-7)

#d_yx 导函数
d_yx=np.polyder(y)
d_yx(-7)