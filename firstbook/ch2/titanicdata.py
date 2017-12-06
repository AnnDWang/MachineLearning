import pandas as pd #pandas是python的数据格式处理类库

#加载泰坦尼克号生存预测数据集
data_train=pd.read_csv("../data/titanic/train.csv")
data_train.info()

data_train.groupby('Survived').count()

data_train.head(3)

print(data_train.head(3))

#数据缺失处理的方法：
#扔掉缺失数据
#按某个统计量补全
#拿模型预测缺失值

#均值填充缺失数据
def set_missing_ages(p_df):
    p_df.loc[(p_df.Age.isnull(),'Age')]=p_df.Age.dropna().mean()
    return p_df

df=set_missing_ages(data_train)

#归一化数据
import sklearn.preprocessing as preprocessing

scaler=preprocessing.StandardScaler()
df['Age_scaled']=scaler.fit_transform(data_train['Age'])
df['Fare_scaled']=scaler.fit_transform(data_train['Fare'])

#归一化有问题，git没有加载出源码，之后处理

# #处理Cabin特征
# def set_cabin_type(p_df):
#     p_df.loc[(p_df.Cabin.notnull()),'Cabin']="Yes"
#     p_df.loc[(p_df.Cabin.isnull()),'Cabin']="No"
#     return p_df
#
# df=set_cabin_type(df)
#
# dummies_pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
# dummies_pclass.head(3)
#
# dummies_embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
# dummies_embarked.loc[61]
#
# dummies_sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
# dummies_sex.head(3)
#
# #把处理好的数据合并
# df=pd.concat([df,dummies_embarked,dummies_sex,dummies_pclass],axis=1)
#
# #noinspection PyUnresolvedReference
# df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
#
# #选择那些特征作为训练特征
# train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*\|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_df.head(1)
#
# print(train_df.head(1))