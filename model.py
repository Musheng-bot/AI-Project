import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ALPHA = 0.5
EPOCH_NUM = 200
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_channels)          # 直接输出 logit
        )
    def forward(self, x):
        return self.net(x)      # raw logits
def train_model(model, features, labels, loss_fn, lr=1e-3, batch_size=256, epoch_num=EPOCH_NUM):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for epoch in range(1, epoch_num+1):
        total_loss = 0.0
        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            logits = model(feat)
            loss = loss_fn(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epoch_num} | Loss: {total_loss/len(loader.dataset):.4f}")
    torch.save(model.state_dict(), f'{model.__class__.__name__}.pth')
    return model
def test_model(model, features, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            logits = model(feat)
            probs = torch.sigmoid(logits)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item() * feat.size(0)
            total_samples += feat.size(0)
            preds = (probs > ALPHA).float()
            correct += (preds == lbl).sum().item()
            tp += ((preds == 1) & (lbl == 1)).sum().item()
            fp += ((preds == 1) & (lbl == 0)).sum().item()
            fn += ((preds == 0) & (lbl == 1)).sum().item()
    acc = correct / total_samples
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * (prec * rec) / (prec + rec + 1e-8)
    avg_loss = total_loss / total_samples
    print("\n=== 测试结果 ===")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"准确率: {acc:.4f}")
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
    # 训练 MLP
    mlp = MLP(in_channels=X_train.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    mlp = train_model(mlp, X_train, y_train, loss_fn, lr=0.1, epoch_num=EPOCH_NUM)
    # 测试
    test_model(mlp, X_test, y_test)
    # 随机森林（不用标准化）
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
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