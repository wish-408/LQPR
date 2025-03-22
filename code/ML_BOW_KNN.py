from sklearn.feature_extraction.text import CountVectorizer
from static_data import *
from tools import get_formatted_time, load_data
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from memory_profiler import profile
from tools import result_report, get_date_time

def predicte(sentences) : 
    data_path1 = '../data/new_labeled_data1.txt'
    X_train, y_train = load_data(data_path1)
    #从本地文件加载vectorizer对象
    vectorizer = joblib.load('vectorizer.pkl')
    
    # 转换测试数据
    X_train_bow = vectorizer.transform(X_train)
     # 创建 KNN 分类器并训练模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bow, y_train)

    sentence_bow = vectorizer.transform(sentences)
    # 使用训练好的模型进行预测
    predicte_labels = knn.predict(sentence_bow)
    return predicte_labels
    
# @profile
def ML_BOW_KNN(train_data_path, test_data_path, result_dir, save_name):

    X_train, y_train = load_data(train_data_path)
    X_test, real_labels = load_data(test_data_path)

    #从本地文件加载vectorizer对象
    vectorizer = joblib.load('vectorizer.pkl')

    X_train_bow = vectorizer.transform(X_train)

    # 转换测试数据
    X_test_bow = vectorizer.transform(X_test)

    # 创建 KNN 分类器并训练模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bow, y_train)

    # 使用训练好的模型进行预测
    predicte_labels = knn.predict(X_test_bow)

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("BoW/KNN", save_path, predicte_labels, real_labels)