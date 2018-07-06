import numpy as np

# ID3 基于信息增益

def getInformationEntropy(dataset):
    y=dataset[-1]
    labels={}
    # 判断分类以及每个分类的数目
    for l in y:
        if l not in labels:
            labels[l]=1
        else:
            labels[l]=labels[l]+1
    # 计算信息熵
    total=0.0
    num=len(y)
    if num!=0:
        for l in labels:
            p=labels[l]/num
            if p!=0:
                total=total-p*np.log2(p)
    return total

# 计算某个属性的信息增益
def getGain(dataset):
    # 获取总的增益
    total_entropy=getInformationEntropy(dataset)
    m,n=np.shape(dataset)
    feature_gain_dict={}
    # m 表示了特征数目, i表示当前是第几个特征
    for i in range(1,m):
        # 第i个特征，第一列是序号，排除在外
        temp_total_num=0
        temp_features_dict={}
        for data in dataset:
            if data[i][1] in temp_features_dict:
                temp_features_dict[data[i][1]].append(data[i])
            else:
                temp_features_dict[data[i][1]]=[data[i]]
        temp_features_entropy_num_dict={}
        for j in temp_features_dict:
            temp_features_entropy_num_dict[j]=[len(temp_features_dict[j]),getInformationEntropy(temp_features_dict[j])]


        a=1
    return feature_gain_dict


dataset=np.loadtxt(open("xigua3.txt", encoding='utf8'),delimiter=',')

feature_dict=getGain(dataset)



