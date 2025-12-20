import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def prepare_data():
    df = pd.read_csv(
        'data/train.csv',
        usecols=[
            'bmi',
            'waist_to_hip_ratio',
            'cholesterol_total',
            'triglycerides',
            'family_history_diabetes',
            'diagnosed_diabetes'
        ]
    )
    df = df.fillna(df.mean())
    X = df.drop('diagnosed_diabetes', axis=1).values.astype(np.float32)
    y = df['diagnosed_diabetes'].values.astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.ravel()
    )
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test), torch.tensor(y_test))

models = [
    'Random Forest',
    'Logistic Regression',
    'SVM',
    'KNN',
    'Decision Tree',
    'Naive Bayes',
    'Neural Network'
]

def rf_train(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=5):
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    rf_pred = rf_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), rf_pred)
    precision = precision_score(y_test.numpy(), rf_pred)
    recall = recall_score(y_test.numpy(), rf_pred)
    f1 = f1_score(y_test.numpy(), rf_pred)
    print("\n=== 随机森林结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return rf_clf, accuracy, precision, recall, f1

def Logistic_regression_train(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    lr_pred = lr_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), lr_pred)
    precision = precision_score(y_test.numpy(), lr_pred)
    recall = recall_score(y_test.numpy(), lr_pred)
    f1 = f1_score(y_test.numpy(), lr_pred)
    print("\n=== 逻辑回归结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return lr_clf, accuracy, precision, recall, f1

def svm_train(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC

    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    svm_pred = svm_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), svm_pred)
    precision = precision_score(y_test.numpy(), svm_pred)
    recall = recall_score(y_test.numpy(), svm_pred)
    f1 = f1_score(y_test.numpy(), svm_pred)
    print("\n=== 支持向量机结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return svm_clf, accuracy, precision, recall, f1

def decision_tree_train(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier

    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    dt_pred = dt_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), dt_pred)
    precision = precision_score(y_test.numpy(), dt_pred)
    recall = recall_score(y_test.numpy(), dt_pred)
    f1 = f1_score(y_test.numpy(), dt_pred)
    print("\n=== 决策树结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return dt_clf, accuracy, precision, recall, f1

def knn_train(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier

    # 离散数据，使用曼哈顿距离
    # weights选择'distance'关注近邻中距离较近的点，以提高分类效果
    knn_clf = KNeighborsClassifier(n_neighbors=200, weights='distance', metric='manhattan', n_jobs=-1)
    knn_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    knn_pred = knn_clf.predict(X_test.numpy())
    correct_proba, convince_proba = get_probabilities(knn_clf, X_test, y_test)
        
    accuracy = accuracy_score(y_test.numpy(), knn_pred)
    precision = precision_score(y_test.numpy(), knn_pred)
    recall = recall_score(y_test.numpy(), knn_pred)
    f1 = f1_score(y_test.numpy(), knn_pred)
    print("\n=== KNN结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"预测正确的概率平均: {correct_proba.mean():.4f}")
    print(f"预测可信度的概率平均: {convince_proba.mean():.4f}")

    return knn_clf, accuracy, precision, recall, f1

def naive_bayes_train(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    nb_clf = GaussianNB()
    nb_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    nb_pred = nb_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), nb_pred)
    precision = precision_score(y_test.numpy(), nb_pred)
    recall = recall_score(y_test.numpy(), nb_pred)
    f1 = f1_score(y_test.numpy(), nb_pred)
    print("\n=== 朴素贝叶斯结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return nb_clf, accuracy, precision, recall, f1

def neural_network_train(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier

    nn_clf = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    nn_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    nn_pred = nn_clf.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), nn_pred)
    precision = precision_score(y_test.numpy(), nn_pred)
    recall = recall_score(y_test.numpy(), nn_pred)
    f1 = f1_score(y_test.numpy(), nn_pred)
    print("\n=== 神经网络结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    return nn_clf, accuracy, precision, recall, f1

# 预测正确的概率， 预测可信度的概率
def get_probabilities(model, X_test, y_test):
    # 仅保留核心逻辑：提取真实标签对应概率
    X_test_np = X_test.numpy() if hasattr(X_test, 'numpy') else X_test
    y_test_np = y_test.numpy().astype(int) if hasattr(y_test, 'numpy') else y_test
    y_pred = model.predict(X_test_np)
    y_pred_np = np.array(y_pred).astype(int)
    
    proba = model.predict_proba(X_test_np)
    correct_proba = [proba[i, y_test_np[i]] for i in range(len(y_test_np))]
    convince_proba = [proba[i, y_pred_np[i]] for i in range(len(y_pred_np))]
    return np.array(correct_proba), np.array(convince_proba)
    

train_functions = {
    'Random Forest': rf_train,
    'Logistic Regression': Logistic_regression_train,
    'SVM': svm_train,
    'Decision Tree': decision_tree_train,
    'KNN': knn_train,
    'Naive Bayes': naive_bayes_train,
    'Neural Network': neural_network_train
}
def plot_feature_importance(model, features):
    imp = model.feature_importances_
    features = [
        'bmi',
        'waist_to_hip_ratio',
        'cholesterol_total',
        'triglycerides',
        'family_history_diabetes'
    ]
    plt.figure(figsize=(8, 6))
    plt.barh(features, imp, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("RandomForest Feature Importance")
    plt.tight_layout()
    plt.show()
    print("\n=== 特征重要性 ===")
    for feature, importance in zip(features, imp):
        print(f"{feature}: {importance:.4f}")

train_models = [
    # 'Random Forest',
    # 'Logistic Regression',
    # 'SVM',
    'KNN',
    # 'Decision Tree',
    # 'Naive Bayes',
    # 'Neural Network'
]
def main():
    X_train, y_train, X_test, y_test = prepare_data()
    for model_name in train_models:
        train_func = train_functions[model_name]
        model, accuracy, precision, recall, f1 = train_func(X_train, y_train, X_test, y_test)
        with open('model_results.txt', 'a') as f:
            f.write(f"{model_name} Results:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
        if model_name == 'Random Forest':
            plot_feature_importance(model, [
                'bmi',
                'waist_to_hip_ratio',
                'cholesterol_total',
                'triglycerides',
                'family_history_diabetes'
            ])

    

if __name__ == "__main__":
    main()