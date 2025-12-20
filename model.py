import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ALPHA = 0.5
EPOCH_NUM = 200


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

    # 先把缺失值填完
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

def main():
    X_train, y_train, X_test, y_test = prepare_data()
    
    # 随机森林（不用标准化）
    rf_clf = RandomForestClassifier(
        n_estimators=20, max_depth=5, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train.numpy(), y_train.numpy().ravel())
    rf_pred = rf_clf.predict(X_test.numpy())
    print("\n=== 随机森林结果 ===")
    print(f"准确率: {accuracy_score(y_test.numpy(), rf_pred):.4f}")
    print(f"精确率: {precision_score(y_test.numpy(), rf_pred):.4f}")
    print(f"召回率: {recall_score(y_test.numpy(), rf_pred):.4f}")
    print(f"F1 分数: {f1_score(y_test.numpy(), rf_pred):.4f}")

if __name__ == "__main__":
    main()