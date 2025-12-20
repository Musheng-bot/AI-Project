import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

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
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.ravel()
    )
    print(f"Train x : \n{X_train[:5]}")
    print(f"Train y : \n{y_train[:5]}")
    print(f"Test x : \n{X_test[:5]}")
    print(f"Test y : \n{y_test[:5]}")
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test), torch.tensor(y_test))



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


models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    # 'SVM': SVC(kernel='linear', probability=True, random_state=42),
    # 'KNN': KNeighborsClassifier(n_neighbors=200, weights='distance', metric='manhattan', n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=3, criterion='entropy'),
    # 'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}


def train_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train.numpy(), y_train.numpy().ravel())
    nb_pred = model.predict(X_test.numpy())

    accuracy = accuracy_score(y_test.numpy(), nb_pred)
    precision = precision_score(y_test.numpy(), nb_pred)
    recall = recall_score(y_test.numpy(), nb_pred)
    f1 = f1_score(y_test.numpy(), nb_pred)
    correct_proba, convince_proba = get_probabilities(model, X_test, y_test)
    print(f"\n=== {model_name} 结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"预测正确的概率平均: {correct_proba.mean():.4f}")
    print(f"预测可信度的概率平均: {convince_proba.mean():.4f}")
    return model, accuracy, precision, recall, f1, correct_proba.mean(), convince_proba.mean()

def n_neighbors_analysis():
    for k in range(5, 305, 10):
        yield k

def main():
    X_train, y_train, X_test, y_test = prepare_data()
    for model_name in models:
        model = models[model_name]
        model, accuracy, precision, recall, f1, correct_proba_mean, convince_proba_mean = train_model(model_name, model, X_train, y_train, X_test, y_test)
        with open('model_results.txt', 'a') as f:
            f.write(f"{model_name} Results:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Correct Probability Mean: {correct_proba_mean:.4f}\n")
            f.write(f"Convince Probability Mean: {convince_proba_mean:.4f}\n\n")


    

if __name__ == "__main__":
    main()