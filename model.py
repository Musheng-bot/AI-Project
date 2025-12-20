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

def train(X_train, y_train, X_test, y_test, n_estimators, max_depth):
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

def main():
    X_train, y_train, X_test, y_test = prepare_data()
    model, *_ = train(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=5)

    imp = pd.Series(model.feature_importances_, index=[
        'bmi',
        'waist_to_hip_ratio',
        'cholesterol_total',
        'triglycerides',
        'family_history_diabetes'
    ]).sort_values(ascending=True)
    imp.plot.barh()
    plt.title("RandomForest Feature Importance")
    plt.tight_layout()
    plt.show()
    print("\n=== 特征重要性 ===")
    for feature, importance in imp.items():
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()