# -*- coding: utf-8 -*
import numpy as np
from abupy import ABuSymbolPd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

def _gen_another_word_price(kl_another_word):
    """
    生成股票在另一个世界中的价格
    :param kl_another_word:
    :return:
    """
    for ind in np.arange(2,kl_another_word.shape[0]):
        # 前天数据
        bf_yesterday = kl_another_word.iloc[ind-2]
        # 昨天数据
        yesterday = kl_another_word.iloc[ind -1]
        # 今天数据
        today = kl_another_word.iloc[ind]
        # 生成今天的收盘价格
        kl_another_word.close[ind] = _gen_another_word_price_rule(
            yesterday.close,yesterday.volume,
            bf_yesterday.close,bf_yesterday.volume,
            today.volume,today.date_week
        )

def _gen_another_word_price_rule(
        yesterday_close,yesterday_volume,bf_yesterday_close, bf_yesterday_volume,today_volume,date_week):
    """
    通过前天的收盘量价、昨天的收盘量价和今天的量，构建另一个世界中的价格模型
    :param yesterday_close:
    :param yesterday_volume:
    :param bf_yesterday_close:
    :param bf_yesterday_volume:
    :param today_volume:
    :param date_week:
    :return:
    """
    # 昨天的收盘价格与前天的收盘价格差
    price_change=yesterday_close-bf_yesterday_close
    # 昨天成交量与前天成交量的量差
    volume_change = yesterday_volume-bf_yesterday_volume
    # 如果量和价格变动一直，则今天价格上涨，否则下跌
    # 即量价齐涨-》涨，量价齐跌-》涨，量价不一致-》跌
    sign = 1.0 if price_change *volume_change >0 else -1.0
    # 针对sign生成噪音，噪音生效的先决条件是今天的量是这三天最大的
    gen_noise = today_volume>np.max([yesterday_volume,bf_yesterday_volume])

    #如果量是这三天最大，且是周五，下跌
    if gen_noise and date_week==4:
        sign=-1.0
    #如果量是这三天最大，且是周一，上涨
    elif gen_noise and date_week==0:
        sign=1.0
    #今天的涨跌幅度基础是price_change(昨天和前天的价格变动)
    price_base=abs(price_change)
    #今天的涨跌幅度变动因素：量比
    #“今天的成交量/昨天的成交量”和“今天的成交量/前天的成交量”的均值
    price_factor = np.mean([today_volume/yesterday_volume,today_volume/bf_yesterday_volume])

    if abs(price_base * price_factor)<yesterday_close*0.10:
        # 如果量比 * price_base 没有超过10%。今天价格计算
        today_price = yesterday_close+sign*price_base*price_factor
    else:
        # 如果涨跌幅超过10%，限制上限、下限为10%
        today_price=yesterday_close+sign*yesterday_close*0.10
    return today_price

def change_real_to_another_word(symbol):
    """
    将原始真正的股票数据价格只保留前两天数据，成交量，周几列完全保留，价格列其他数据使用_gen_another_word_price,变成另一个世界价格
    :param symbol:
    :return:
    """
    kl_pd=ABuSymbolPd.make_kl_df(symbol)
    if kl_pd is not None:
        #原始股票数据只保留价格、周几和成交量
        kl_fairy_tale=kl_pd.filter(['close','date_week','volume'])
        #只保留头两天的原始交易收盘价格，其他都赋予nan
        kl_fairy_tale['close'][2:]=np.nan
        #将其他nan价格变成童话世界的价格需使用_gen_another_word_price
        _gen_another_word_price(kl_fairy_tale)
        return kl_fairy_tale


def gen_fairy_tale_feature(kl_another_word):
    """
    构建特征模型函数
    生成的dataframe有收盘价、周几、成交量列
    :param kl_another_word:
    :return:
    """
    # y值使用close.pct_change，即涨跌幅度
    kl_another_word['regress_y']=kl_another_word.close.pct_change()
    # 前天收盘价格
    kl_another_word['bf_yesterday_close']=0
    # 昨天收盘价格
    kl_another_word['yesterday_close']=0
    # 昨天收盘成交量
    kl_another_word['yesterday_volume']=0
    # 前天收盘成交量
    kl_another_word['bf_yesterday_volume']=0
    # 对齐特征，即前天收盘价与今天的收盘价错开两个时间单位 ，[2:]=[:-2]
    kl_another_word['bf_yesterday_close'][2:]=kl_another_word['close'][:-2]
    # 对齐特征，前天成交量
    kl_another_word['bf_yesterday_volume'][2:]=kl_another_word['volume'][:-2]
    # 对齐特征，昨天收盘价与今天收盘价错一个时间单位，[1:]=[:-1]
    kl_another_word['yesterday_close'][1:]=kl_another_word['close'][:-1]
    # 对齐特征，昨天成交量
    kl_another_word['yesterday_volume'][1:]=kl_another_word['volume'][:-1]
    # 特征1：价格差
    kl_another_word['feature_price_change']=kl_another_word['yesterday_close']-kl_another_word['bf_yesterday_close']
    # 特征2：成交量差
    kl_another_word['feature_volume_change']=kl_another_word['yesterday_volume']-kl_another_word['bf_yesterday_volume']
    # 特征3：涨跌sign
    kl_another_word['feature_sign']=np.sign(kl_another_word['feature_price_change']*kl_another_word['feature_volume_change'])
    # 特征4：周几
    kl_another_word['feature_date_week']=kl_another_word['date_week']

    """
    构建噪音特征，因为我们不可能全部分析正确真实的特征因素，这里引入噪音特征
    """
    # 成交量积
    kl_another_word['feature_volume_noise']=kl_another_word['yesterday_volume']*kl_another_word['bf_yesterday_volume']
    # 价格成绩
    kl_another_word['feature_price_noise']=kl_another_word['yesterday_close']*kl_another_word['bf_yesterday_close']
    # 将数据标准化
    scaler=preprocessing.StandardScaler()
    kl_another_word['feature_price_change']=scaler.fit_transform(kl_another_word['feature_price_change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_change']=scaler.fit_transform(kl_another_word['feature_volume_change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_noise']=scaler.fit_transform(kl_another_word['feature_volume_noise'].values.reshape(-1, 1))
    kl_another_word['feature_price_noise']=scaler.fit_transform(kl_another_word['feature_price_noise'].values.reshape(-1, 1))
    # 只筛选feature_开头的特征和regress_y，抛弃前两天数据，即[2:]
    kl_fairy_tale_feature=kl_another_word.filter(regex='regress_y|feature_*')[2:]
    return kl_fairy_tale_feature

#下面代码选定一些股票，并使用change_real_to_another_word函数构建价格走势，从输出看数据走势。
#选择若干美股股票
choice_symbols=['usNOAH','  usSFUN','usBIDU','usAAPL','usGOOG','usTSLA','usWUBA','usVIPS']

another_word_dict={}
real_dict={}

for symbol in choice_symbols:
    # 通话世界的股票走势字典
    another_word_dict[symbol]=change_real_to_another_word(symbol)
    #真实世界的股票走势字典，这里不考虑运行效率问题
    real_dict[symbol]=ABuSymbolPd.make_kl_df(symbol)

fairy_tale_feature=None
for symbol in another_word_dict:
    # 首先拿出数据对应的走势数据
    kl_another_word=another_word_dict[symbol]
    print kl_another_word
    # 通过走势数据生成训练集特征
    kl_feature=gen_fairy_tale_feature(kl_another_word)
    # 将每个股票的特征数据都拼接起来，形成训练集
    fairy_tale_feature=kl_feature if fairy_tale_feature is None else fairy_tale_feature.append(kl_feature)

print fairy_tale_feature.shape

# dataframe -> matrix
feature_np=fairy_tale_feature.as_matrix()
# X特征矩阵
train_x=feature_np[:,1:]
# 回归的连续y值
train_y_regress=feature_np[:,0]
# 分类训练的离散值y，之后分类计数使用
train_y_classification=np.where(train_y_regress>0,1,0)

print train_x[:5],train_y_regress[:5],train_y_classification[:5]

def gen_feature_from_symbol(symbol):
    """封装一个由symbol转换为特征矩阵的序列函数"""
    # 真实世界走势数据转换到通话世界
    kl_another_word=change_real_to_another_word(symbol)
    # 由走势转换为特征dataframe
    kl_another_word_feature_test=gen_fairy_tale_feature(kl_another_word)
    # 转换为matrix
    feature_np_test=kl_another_word_feature_test.as_matrix()
    # 从matrix抽取y回归
    test_y_regress=feature_np_test[:,0]
    # y回归到y分类
    test_y_classification=np.where(test_y_regress>0,1,0)
    # 从matrix抽取X特征矩阵
    test_x=feature_np_test[:,1:]
    return test_x,test_y_regress,test_y_classification,kl_another_word_feature_test

test_x,test_y_regress,test_y_classification,kl_another_word_feature_test=gen_feature_from_symbol('usFB')

# 下面使用scikit-learn的线性回归模块预测股价涨跌幅度
from sklearn.linear_model import  LinearRegression
from sklearn import cross_validation

def regress_process(estimator,train_x,train_y_regress,test_x,test_y_regress):
    # 训练训练集数据
    estimator.fit(train_x,train_y_regress)
    # 使用训练好的模型预测测试集对应的y，即根据usFB的走向特征预测股价
    test_y_predict_regress=estimator.predict(test_x)

    # 绘制usFB实际股价
    plt.plot(test_y_regress.cumsum())
    # 绘制通过模型预测的usFB股价
    plt.plot(test_y_predict_regress.cumsum())

    # 针对训练集数据进行交叉验证
    scores=cross_validation.cross_val_score(estimator,train_x,train_y_regress,cv=10,scoring='mean_squared_error')

    # mse开方-》rmse
    mean_sc=np.mean(np.sqrt(-scores))

    print('RMSE: '+str(mean_sc))

#实例化线性回归对象estimator
estimator=LinearRegression()

#将回归模型对象，训练集x、训练集连续y值、测试集x、测试集连续y值传入
regress_process(estimator,train_x,train_y_regress,test_x,test_y_regress)

# 使用abu量化中的ABuMLExecute
from abupy import ABuMLExecute
ABuMLExecute.plot_learning_curve(estimator,train_x,train_y_regress,cv=10)


# 多项式回归
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#pipeline套上degree=3+LinearRegression
estimator=make_pipeline(PolynomialFeatures(degree=3),LinearRegression())
#继续使用regress_process
regress_process(estimator,train_x,train_y_regress,test_x,test_y_regress)

#Adaboost：
from sklearn.ensemble import AdaBoostRegressor

estimator=AdaBoostRegressor(n_estimators=100)
regress_process(estimator,train_x,train_y_regress,test_x,test_y_regress)

# 随机森林代码
from sklearn.ensemble import RandomForestRegressor
estimator=RandomForestRegressor(n_estimators=100)
regress_process(estimator,train_x,train_y_regress,test_x,test_y_regress)

# 逻辑分类回归
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
def classification_process(estimator,train_x,train_y_classification,test_x,test_y_classification):
    # 训练数据，这里分类要用y_classification
    estimator.fit(train_x,train_y_classification)
    # 使用训练好的分类模型预测测试集对应的y，即根据usFB的走势预测涨跌
    test_y_predict_classification=estimator.predict(test_x)
    print("{} accuracy={:.2f}".format(estimator.__class__.__name__,metrics.accuracy_score(test_y_classification,test_y_predict_classification)))

    # 针对训练集数据做交叉验证scoring='accuracy'
    scores=cross_validation.cross_val_score(estimator,train_x,train_y_classification,cv=10,scoring='accuracy')

    mean_sc=np.mean(scores)
    print('cross validation accuracy mean:{:.2f}'.format(mean_sc))

estimator=LogisticRegression(C=1.0,penalty='11',tol=1e-6)
classification_process(estimator,train_x,train_y_classification,test_x,test_y_classification)

#使用随机森林来运行看效果
from sklearn.ensemble import RandomForestClassifier

estimator=RandomForestClassifier(n_estimators=100)
classification_process(estimator,train_x,train_y_classification,test_x,test_y_classification)

#使用cross_validation模块下的函数切分训练测试集，将测试集设置为0.5
from sklearn.cross_validation import train_test_split
def train_test_split_xy(estimator,x,y,test_size=0.5,random_state=0):
    # 通过train_test_split将原始训练集随机切割为新的训练集与测试集
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=test_size,random_state=random_state)

    print(x.shape,y.shape)
    print(train_x.shape,train_y.shape)
    print(test_x.shape, test_y.shape)

    clf=estimator.fit(train_x,train_y)
    predictions=clf.predict(test_x)

    #度量准确率
    print("accuracy=%.2f"%(metrics.accuracy_score(test_y,predictions)))

    #度量查准率
    print("precision_score=%.2f"%(metrics.precision_score(test_y,predictions)))

    #度量回收率
    print("recall_score=%.2f" % (metrics.recall_score(test_y, predictions)))

    return test_y,predictions

test_y,predictions=train_test_split_xy(estimator,train_x,train_y_classification)

#查看ROC曲线
ABuMLExecute.plot_roc_estimator(estimator,train_x,train_y_classification)

# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import os

estimator=DecisionTreeClassifier(max_depth=2,random_state=1)

def graphviz_tree(estimator,features,x,y):
    if not hasattr(estimator,'tree_'):
        print('only tree can graphviz!')
        return

    estimator.fit(x,y)
    # 将决策模型到处成graphviz.dot模型
    tree.export_graphviz(estimator.tree_,out_file='graphviz.dot',feature_names=features)
    #通过dot将模型绘制成决策图，保存png
    os.system("dot -T png graphviz.dot -o graphviz.png")

#这里会用到特征的名称列fairy_tale_feature.columns[1:]
graphviz_tree(estimator,fairy_tale_feature.columns[1:],train_x,train_y_classification)

import pandas as pd

#下面使用RandomForestClassifier作为实例分类器来查看特征的重要性
def importances_coef_pd(estimator):
    if hasattr(estimator,'feature_importances_'):
        #有feature_importances_的通过sort_values排序
        return pd.DataFrame(
            {'feature':list(fairy_tale_feature.columns[1:]),
             'importance':estimator.feature_importances_}
        ).sort_values('importance')
    #有coef的通过coef排序
    elif hasattr(estimator,'coef_'):
        return pd.DataFrame(
            {'columns':list(fairy_tale_feature.columns[1:]),'coef':list(estimator.coef_.T)}
        ).sort_values('coef')
    else:
        print('estimator not hasattr feature_importances or coef_ !')
#使用随机森林分类器
estimator=RandomForestClassifier(n_estimators=100)
#训练数据模型
estimator.fit(train_x,train_y_classification)
#对训练后的模型特征的重要程度进行判定，重要程度有小到大
importances_coef_pd(estimator)

from sklearn.feature_selection import  RFE

def feature_selection(estimator,x,y):
    selector=RFE(estimator)
    selector.fit(x,y)
    print('RFE selection')
    print(pd.DataFrame(
        {'support':selector.support_,'ranking':selector.ranking_},
        index=fairy_tale_feature.columns[1:]
    ))

feature_selection(estimator,train_x,train_y_classification)

# 3.3
from abupy import AbuML
# 通过X ，Y矩阵和特征的DataFrame对象醉成AbuML
ml=AbuML(train_x,train_y_classification,fairy_tale_feature)
# 使用随机森林作为分类器
_=ml.estimator.random_forest_classifier()

# 交织验证结果的正确率
ml.cross_val_accuracy_score()
# 特征的选择
ml.feature_selection()

abupy.env.g_enable_ml_feature=True
abupy.env.g_enable_train_test_split=True

# 初始化资金200万元
read_cash=2000000
# 每笔交易的买入基数资金设置为万分之15
abupy.beta.atr.g_atr_pos_base=0.0015

# 使用run_loop_back运行策略进行全市场回测：
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import abu
# 设置选股因子，None为不适用选股因子
stock_pickers=None
# 买入因子使用向上突破因子
buy_factors=[{'xd':60,'class':AbuFactorBuyBreak},{'xd':42,'class':AbuFactorBuyBreak}]

#卖出因子
sell_factores=[{'stop_loss_n':1.0,'stop_win_n':3.0,'class':AbuFactorAtrNStop},
               {'class':AbuFactorPreAtrNStop,'pre_atr_n':1.5},
               {'class':AbuFactorCloseAtrNStop,'close_atr_n':1.5}]

# 全市场
choise_symbols=None
# 使用run_loop_back运行策略，5年历史数据回测
abu_result_tuple,kl_pd_manger=abu.run_loop_back(read_cash,buy_factors,sell_factores,stock_pickers,choice_symbols=choise_symbols,n_folds=5)

from abupy import AbuMetricsBase

metrics=AbuMetricsBase(*abu_result_tuple)
metrics.fit_metrics()
metrics.plot_returns_cmp(only_show_returns=True)

abupy.evn.g_enable_train_test_split=False

#使用刚才切割股票池中的测试集symbols
abupy.env.g_enable_last_split_test=True
read_cash=2000000
abupy.beta.attr.g_atr_pos_base=0.015
choice_symbols=None
abu_result_tuple_test,_=abu.run_loop_back(read_cash,buy_factors,sell_factores,stock_pickers,choice_symbols=choise_symbols,n_folds=5)

metrics=AbuMetricsBase(*abu_result_tuple_test)
metrics.fit_metrics()
metrics.plot_returns_cmp(only_show_returns=True)

abu_result_tuple.orders_pd.columns

#使用几个综合特征来训练模型
from abupy import  AbuUmpMainMul

mul=AbuUmpMainMul.UmpMulFiter(orders_pd=abu_result_tuple.orders_pd,scaler=False)

mul.df.head()

mul().cross_val_accuracy_score()

# 随机森林
mul().estimator.random_forest_classifier()
mul().cross_val_accuracy_score()

#使用历史拟合角度特征实验
from abupy import AbuUmpMainDeg
deg=AbuUmpMainDeg.UmpDegFilter(orders_pd=abu_result_tuple.orders_pd)
# 分类器使用adaboost
deg().estimator.adaboost_classifier()
deg.df.head()

deg().cross_val_accuracy_score()

# 混淆矩阵分布
deg().train_test_split_xy()

#使用更多特征
from abupy import  AbuUmpMainFull
full=AbuUmpMainFull.UmpFullFilter(orders_pd=abu_result_tuple.orders_pd)
#继续使用adaboost
full().estimator.adaboost_classifier()
#查看full所有特征名称
full.df.columns

#full交叉验证
full().cross_val_accuracy_score()