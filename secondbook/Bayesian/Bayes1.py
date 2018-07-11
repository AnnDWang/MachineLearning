# # -*- coding:utf-8 -*-
import numpy as np

# 拉普拉斯贝叶斯的实现
# 参考西瓜书 第七章
# 根据数据，计算出来的方差的结果有点问题，导致最终结果不太正确

# 获取数据的方法
def readFile(filename,split_str):
    # files = open(filename)
    # 如果读取不成功试一下
    files = open(filename, "r", encoding="gbk")
    data = []
    for line in files.readlines():
        item = line.strip().split(split_str)
        data.append(item)
    return data

dataset=readFile("xigua3.csv",',')

# 计算先验概率
# 判断类别数目
labels_dic={}

for data in dataset:
    if data[-1] not in labels_dic:
        labels_dic[data[-1]]=1
    else:
        labels_dic[data[-1]]=labels_dic[data[-1]]+1

# 先验概率
p_c_dic={}
total_num=len(dataset)
total_class=len(labels_dic)
for l in labels_dic:
    temp=labels_dic[l]+1
    p_c_dic[l]=temp/(total_num+total_class)

# 计算每个属性的类先验概率
total_features_dic={}
for i in range(1,7):
    temp_feature={}
    for data in dataset:
        if data[i] not in temp_feature:
            temp_feature[data[i]]=[data[-1]]
        else:
            temp_feature[data[i]].append(data[-1])
    total_features_dic[i]=temp_feature

# 计算离散属性概率
total_features_p_dic={}

for feature in total_features_dic:
    temp_features=total_features_dic[feature]
    total_features_p_dic[feature]={}

    feature_values_num=len(temp_features) # Ni

    for cur_feature in temp_features:
        cur_class = temp_features[cur_feature]

        temp_classes={}
        for l in labels_dic:
            cur_label=labels_dic[l] # dc
            temp_classes[l]=0
            for c in cur_class:
                if c==l:
                    temp_classes[l]+=1
        temp1={}
        for l in temp_classes:
            temp1[l] = (temp_classes[l] + 1) / (labels_dic[l] + feature_values_num)

        total_features_p_dic[feature][cur_feature] = temp1
    # for l in labels_dic:
    #     cur_label=labels_dic[l] # dc
    #     temp1 = {}
    #     for cur_feature in temp_features:
    #         cur_class=temp_features[cur_feature]
    #         c_t_num=0
    #         for c in cur_class:
    #             if c==l:
    #                 c_t_num+=1
    #         temp1[l]=(c_t_num+1)/(cur_label+feature_values_num)
    #
    #         total_features_p_dic[feature][cur_feature]=temp1
# 计算连续属性的均值和方差
total_lianxu_features_p_dic={}
for i in range(7,9):
    temp={}
    for l in labels_dic:
        temp[l]=[]
        for data in dataset:
            if data[-1]==l:
                temp[l].append(float(data[i]))
    # 均值和方差
    mean_var={}
    for l in temp:
        nums=temp[l]
        mean=np.mean(nums)
        var_value=np.var(nums)
        mean_var[l]=[mean,var_value]
    total_lianxu_features_p_dic[i]=mean_var

def predict(data):
    # 判断类别
    temp=[]
    for l in labels_dic:
        pc=1
        for i in range(1,7):
            # 获取该属性
            pc=pc*total_features_p_dic[i][data[i]][l]
        for i in range(7,9):
            mean=total_lianxu_features_p_dic[i][l][0]
            var_value=total_lianxu_features_p_dic[i][l][1]
            cur_p=1/(np.sqrt(2*np.pi)*var_value)*np.exp(-(data[i]-mean)*(data[i]-mean)/(var_value*var_value))
            pc=pc*cur_p
        temp.append((pc,l))
    temp.sort(reverse=True)
    print('预测分类为：'+temp[0][1])

predict(['1','青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0,460])
a=1