import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


pd.set_option('display.max_rows', 100)  # 设置最大显示100行
pd.set_option('display.max_columns', 100)  # 设置最大显示100列

data = pd.read_csv("D:\\AQG\\train.csv")
print(data.head(10))    # 查看前10行

# print(data.describe())
# data.info()   # 查看data的信息

print(data.isnull().sum().sort_values(ascending=False))    # 检查缺失值情况
msno.matrix(data)    # 以图表的形式查看缺失情况
plt.show()

# 查看gender特征有哪些值
# print(data['gender'].unique())
data['gender'] = data['gender'].fillna('M')
# 进行独热编码
data.loc[data['gender'] == 'M', 'gender'] = 0     # 另gender等于male那行的gender值为0
data.loc[data['gender'] == 'F', 'gender'] = 1     # 另gender等于female那行的gender值为1
# print(data['gender'].unique())

# print(data['difficulty_level'].unique())
data['difficulty_level'] = data['difficulty_level'].fillna('easy')
data.loc[data['difficulty_level'] == 'intermediate', 'difficulty_level'] = 0     # 另gender等于male那行的gender值为0
data.loc[data['difficulty_level'] == 'easy', 'difficulty_level'] = 1
data.loc[data['difficulty_level'] == 'hard', 'difficulty_level'] = 2
data.loc[data['difficulty_level'] == 'vary hard', 'difficulty_level'] = 3

# print(data['test_type'].unique())
data['test_type'] = data['test_type'].fillna('offline')
data.loc[data['test_type'] == 'offline', 'test_type'] = 0     # 另gender等于male那行的gender值为0
data.loc[data['test_type'] == 'online', 'test_type'] = 1     # 另gender等于female那行的gender值为1

# print(data['education'].unique())
data['education'] = data['education'].fillna('High School Diploma')
data.loc[data['education'] == 'Matriculation', 'education'] = 0     # 另gender等于male那行的gender值为0
data.loc[data['education'] == 'High School Diploma', 'education'] = 1     # 另gender等于female那行的gender值为1
data.loc[data['education'] == 'Bachelors', 'education'] = 2
data.loc[data['education'] == 'Masters', 'education'] = 3
data.loc[data['education'] == 'No Qualification', 'education'] = 4

# print(data['is_handicapped'].unique())
data['is_handicapped'] = data['is_handicapped'].fillna('N')
data.loc[data['is_handicapped'] == 'N', 'is_handicapped'] = 0     # 另gender等于male那行的gender值为0
data.loc[data['is_handicapped'] == 'Y', 'is_handicapped'] = 1     # 另gender等于female那行的gender值为1

data['trainee_engagement_rating'] = data['trainee_engagement_rating'].fillna(1)
data['total_programs_enrolled'] = data['total_programs_enrolled'].fillna(2)
data['city_tier'] = data['city_tier'].fillna(3)
# data['age'] = data['age'].fillna(data['age'].mean())
data['program_duration'] = data['program_duration'].fillna(data['program_duration'].mean())

# print(data['program_type'].unique())
data['program_type'] = data['program_type'].fillna('Y')
data.loc[data['program_type'] == 'Y', 'program_type'] = 0
data.loc[data['program_type'] == 'T', 'program_type'] = 1
data.loc[data['program_type'] == 'Z', 'program_type'] = 2
data.loc[data['program_type'] == 'V', 'program_type'] = 3
data.loc[data['program_type'] == 'U', 'program_type'] = 4
data.loc[data['program_type'] == 'X', 'program_type'] = 5
data.loc[data['program_type'] == 'S', 'program_type'] = 6
# print(data['program_type'].unique())

data['trainee_engagement_rating'] = data['trainee_engagement_rating'].fillna(1)
data['total_programs_enrolled'] = data['total_programs_enrolled'].fillna(2)
data['city_tier'] = data['city_tier'].fillna(3)
data['age'] = data['age'].fillna(data['age'].mean())
data['program_duration'] = data['program_duration'].fillna(data['program_duration'].mean())

# msno.matrix(data)    # 再次检查缺失值情况
# data.info()
# plt.show()

# 去除无用特征(总数据)
drops = ["id_num", "program_id", "test_id", "trainee_id"]
data = data.drop(drops, axis=1)
# print(data.head(6))

print(data.info())

# 划分出真实值和特征
X = data.loc[:, "program_type":"trainee_engagement_rating"].copy()
y = data.loc[:, "is_pass"].copy()

# 留出法，将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape)

# 特征选择
sc = StandardScaler()
sc.fit(X_train)    # 估算每个特征的平均值和标准差
print(sc.mean_)
print(sc.scale_)
# drops = ["program_duration", "difficulty_level", "education", "total_programs_enrolled", "test_type",
#          "age", "gender"]
# X_train = data.drop(drops, axis=1)
# print(X_train.head(6))
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 查看各特征的得分
predictors = ["program_type","program_duration","test_type", "difficulty_level", "gender","trainee_engagement_rating",
              "total_programs_enrolled", "city_tier", "age", "is_handicapped", "education"]
selector = SelectKBest(f_classif, k=5)     # f_classif：基于方差分析的检验统计f值，根据k个最高分数选择功能
selector.fit(data[predictors], data["is_pass"])

scores = -np.log10(selector.pvalues_ + 1e-5)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# X_train.info()

j = 0
for i in [1, 2, 3, 4, 5, 7, 8]:

    X_train_std = np.delete(X_train_std, i-j, 1)
    X_test_std = np.delete(X_test_std, i-j, 1)
    j += 1

# drops = ["program_duration", "difficulty_level", "education", "total_programs_enrolled", "test_type",
#          "age", "gender"]
# X_train_std = X_train_std.drop(drops, axis=1)
# X_test_std = X_test_std.drop(drops, axis=1)

# X_train_std = np.array(X_train_std)
# X_test_std = np.array(X_test_std)

print(X_train_std.shape, X_test_std.shape, y_train.shape)

clf = Perceptron(eta0=0.8, max_iter=80)

bdt = AdaBoostClassifier(clf, algorithm="SAMME", n_estimators=400, learning_rate=0.8)
bdt.fit(X_train_std, y_train)

print("Score:", bdt.score(X_train_std, y_train))

y_pred = bdt.predict(X_test_std)

# 模型评估
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# 开始预测
test_data = pd.read_csv("D:\\AQG\\test2.csv")

# id_num = test_data.loc[:, "id_num"].copy()
# id_num = np.array(id_num).reshape(-1, 1)
#
# test_data = test_data.drop(drops, axis=1)

test_data.loc[test_data['program_type'] == 'Y', 'program_type'] = 0
test_data.loc[test_data['program_type'] == 'T', 'program_type'] = 1
test_data.loc[test_data['program_type'] == 'Z', 'program_type'] = 2
test_data.loc[test_data['program_type'] == 'V', 'program_type'] = 3
test_data.loc[test_data['program_type'] == 'U', 'program_type'] = 4
test_data.loc[test_data['program_type'] == 'X', 'program_type'] = 5
test_data.loc[test_data['program_type'] == 'S', 'program_type'] = 6

test_data.loc[test_data['test_type'] == 'offline', 'test_type'] = 0
test_data.loc[test_data['test_type'] == 'online', 'test_type'] = 1

test_data.loc[test_data['education'] == 'Matriculation', 'education'] = 0
test_data.loc[test_data['education'] == 'High School Diploma', 'education'] = 1
test_data.loc[test_data['education'] == 'Bachelors', 'education'] = 2
test_data.loc[test_data['education'] == 'Masters', 'education'] = 3
test_data.loc[test_data['education'] == 'No Qualification', 'education'] = 4

test_data.loc[test_data['is_handicapped'] == 'N', 'is_handicapped'] = 0
test_data.loc[test_data['is_handicapped'] == 'Y', 'is_handicapped'] = 1

# test_data['age'] = test_data['age'].fillna(test_data['age'].mean())
test_data['trainee_engagement_rating'] = \
    test_data['trainee_engagement_rating'].fillna(test_data['trainee_engagement_rating'].mean())

id_num = test_data.loc[:, "id_num"].copy()
id_num = np.array(id_num).reshape(-1, 1)


# 去除无用特征(总数据)
drops = ["id_num", "program_id", "test_id", "trainee_id", "program_duration", "difficulty_level", "education",
         "total_programs_enrolled", "test_type", "age", "gender"]
test_data = test_data.drop(drops, axis=1)

test = test_data.loc[:, "program_type":"trainee_engagement_rating"].copy()

# test = np.array(test)


test = np.array(test)

predict = bdt.predict(test)
predict = np.array(predict).reshape(-1, 1)
# print(predict)


# 保存预测结果
res = np.concatenate((id_num, predict), axis=1)
df = pd.DataFrame(res).to_csv('test2_submission.csv', index=0, header=1)
