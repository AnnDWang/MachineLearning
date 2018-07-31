from numpy import *

# 在某个区间范围内随机选择一个整数
# i是第一个alpha的下标，m是所有alpha的数目
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

# 用于在数值太大时对其进行调整
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj



