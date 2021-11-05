from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn import neighbors
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import time
import os


# 利用Pandas自带的read_csv函数读取并转化为DataFrame格式
data = pd.read_csv('rp.csv')

# 为了方便仅选取五个简单的特征
datas = data[['salary','satisfaction_level','last_evaluation','number_project',
            'average_montly_hours','time_spend_company']]
amm=datas['salary']
#打印出前五行查看数据情况
print(datas.head(5))

#查看数据描述
datas['salary'].unique()
pd.Series(datas['salary']).value_counts()
print(datas.describe())
# 画出数据分布直方图
datas.hist(bins=50,figsize=(20,15))
#salary分布直方图
plt.hist(amm,bins=50)
plt.show()


def trans(x):
    if x == datas['salary'].unique()[0]:
        return 0
    if x == datas['salary'].unique()[1]:
        return 1
    if x == datas['salary'].unique()[2]:
        return 2

datas['salary'] = datas['salary'].apply(trans)
for col in datas.columns:
    if col != 'salary':
        sns.boxplot(x='salary', y=col, saturation=0.5, palette='pastel', data=datas)
        plt.title(col)
        plt.show()
        

Y = datas[datas['salary'].isin([0,1,2])][['salary']]
X = datas[datas['salary'].isin([0,1,2])][['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']]

target_names=['LOW','MEDIUM','HIGH']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

warnings.filterwarnings("ignore")

#决策树
start_time1 = time.time()
tree = tree.DecisionTreeClassifier(criterion='entropy')
tree.fit(x_train,y_train)
print("决策树模型训练集的准确率:%.3f"%tree.score(x_train,y_train))
print("决策树测试集的准确率:%.3f"%tree.score(x_test,y_test))
y1_hat=tree.predict(x_test)
print(classification_report(y_test,y1_hat,target_names=target_names))
print('training took %fs!' % (time.time() - start_time1))
print('******************* ovo ********************' )



#朴素贝叶斯
start_time2 = time.time()
bayes=naive_bayes.GaussianNB()
bayes.fit(x_train,y_train)
print("贝叶斯模型训练集的准确率:%.3f"%bayes.score(x_train,y_train))
print("贝叶斯模型测试集的准确率:%.3f"%bayes.score(x_test,y_test))
y2_hat=bayes.predict(x_test)
print(classification_report(y_test,y2_hat,target_names=target_names))
print('training took %fs!' % (time.time() - start_time2))
print('******************* ovo ********************' )

#KNN最近邻
start_time4 = time.time()
start4 = time.perf_counter()
knn = KNeighborsClassifier() 
knn.fit(x_train, y_train) 
knn_y4_predict = knn.predict(x_test)
score=knn.score(x_test,y_test)
print("最近邻模型训练集的准确率:%.3f"%knn.score(x_train,y_train))
print("最近邻模型测试集的准确率:%.3f"%knn.score(x_test,y_test))
y4_hat=knn.predict(x_test)
print(classification_report(y_test,y4_hat,target_names=target_names))
print('training took %fs!' % (time.time() - start_time4))
print('elaspe: {0:.6f}'.format(time.perf_counter()-start4))
print('******************* ovo ********************' )


#BP神经网络
start_time5 = time.time()
start5 = time.perf_counter()
nerve=MLPClassifier()
nerve.fit(x_train,y_train)
print("神经网络模型训练集的准确率:%.3f"%nerve.score(x_train,y_train))
print("神经网络模型测试集的准确率:%.3f"%nerve.score(x_test,y_test))
y5_hat=nerve.predict(x_test)
print(classification_report(y_test,y5_hat,target_names=target_names))
print('training took %fs!' % (time.time() - start_time5))
print('elaspe: {0:.6f}'.format(time.perf_counter()-start5))
print('******************* ovo ********************' )

