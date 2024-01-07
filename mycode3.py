from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
# 1.获取数据集
iris = load_iris()
# 2.数据基本处理
# x_train,x_test,y_train,y_test为训练集特征值、 测试集特征值、 训练集⽬标值、 测试集⽬标值
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# 3.特征工程-标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4.机器学习-KNN
estimator = KNeighborsClassifier(n_neighbors=5)
estimator.fit(x_train, y_train)
# 5.模型评估
# 方法1：直接对比真实值和预测值
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接对比真实值和预测值:\n", y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为:\n", score)
