# 1、导入相应的包
import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

with open('stopword.txt', 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file.readlines()])

# 2、加载数据，进行文本预处理
def preprocess_text(text):
    # 正则匹配，去除特殊字符
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # jieba分词
    words = jieba.lcut(text)
    # 去停用词
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# 加载数据
data = pd.read_csv('filtered_cnews.txt', sep='\t', header=None, names=['label', 'text'])
data['processed_text'] = data['text'].apply(preprocess_text)

# 3、划分数据集
# 类别均衡划分
X = data['processed_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
# 再将测试集分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

# 4、文本向量化以及特征选择
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
ch2 = SelectKBest(chi2, k=10000)  # 选择10000个最佳特征
X_train_chi = ch2.fit_transform(X_train_counts, y_train)

# 5、模型训练与评估
# 朴素贝叶斯
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_chi, y_train)
X_test_counts = vectorizer.transform(X_test)
X_test_chi = ch2.transform(X_test_counts)
y_pred_nb = nb_classifier.predict(X_test_chi)
print("Naive Bayes Classifier:")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# KNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_chi, y_train)
y_pred_knn = knn_classifier.predict(X_test_chi)
print("KNN Classifier:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# GBDT
gbdt_classifier = GradientBoostingClassifier()
gbdt_classifier.fit(X_train_chi, y_train)
y_pred_gbdt = gbdt_classifier.predict(X_test_chi)
print("GBDT Classifier:")
print(confusion_matrix(y_test, y_pred_gbdt))
print(classification_report(y_test, y_pred_gbdt))
