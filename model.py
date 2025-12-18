import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 数据标准化
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

ALPHA = 0.5
EPOCH_NUM = 200

# 逻辑斯蒂回归（二分类专用）
class LGRegression(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# 多层感知机（MLP）二分类
class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, out_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_dim // 2, out_features=out_channels, bias=True),
            nn.Sigmoid(),  # 二分类最终输出Sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_model(
        model: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        loss_fn,
        lr: float = 0.01,
        batch_size: int = 64,
        epoch_num: int = 100
) -> nn.Module:
    """
    训练模型（适配二分类任务）
    :param model: 模型实例
    :param features: 特征张量 [样本数, 特征数]
    :param labels: 标签张量 [样本数, 1] (float类型，0/1)
    :param loss_fn: 损失函数（BCELoss）
    :param lr: 学习率
    :param batch_size: 批次大小
    :param epoch_num: 训练轮数
    :return: 训练好的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 模型准备
    model.train()
    model = model.to(device)

    # 优化器（添加L2正则化）
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Adam比SGD更适合MLP

    # 数据加载器
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # 训练循环
    for epoch in range(epoch_num):
        total_loss: float = 0.0
        batch_count: int = 0

        for feature, label in dataloader:
            # 数据移到设备
            feature = feature.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 前向传播
            output = model(feature)

            # 计算损失
            loss = loss_fn(output, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            total_loss += loss.item()
            batch_count += 1

        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            print(f"Epoch {(epoch + 1)}/{epoch_num} | Average loss: {avg_loss:.4f}")

    return model


def test_model(
        model: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64,
) -> None:
    """
    测试模型（二分类）
    :param model: 训练好的模型
    :param features: 测试特征 [样本数, 特征数]
    :param labels: 测试标签 [样本数, 1] (float类型)
    :param batch_size: 批次大小
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    # 数据加载器
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 损失函数
    loss_fn = nn.BCELoss(reduction='mean')

    with torch.no_grad():
        total_loss: float = 0.0
        total_samples: int = 0
        correct_num: int = 0

        for feature, label in dataloader:
            feature = feature.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 前向传播
            output = model(feature)

            # 计算损失
            loss = loss_fn(output, label)
            total_loss += loss.item() * label.shape[0]
            total_samples += label.shape[0]

            predicted = (output > ALPHA).float()
            correct_num += (predicted == label).sum().item()

        # 计算指标
        accuracy = correct_num / total_samples
        average_loss = total_loss / total_samples

        print("=" * 50)
        print(f"测试结果(设备: {device}): ")
        print(f"平均损失: {average_loss:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 50)


def train_rf(
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2
) -> tuple:
    """
    训练随机森林（sklearn版本）
    :param features: 特征数组 [样本数, 特征数]
    :param labels: 标签数组 [样本数] (int类型)
    :param test_size: 测试集比例
    :return: 训练好的分类器、测试特征、测试标签
    """
    # 划分数据集
    feature_train, feature_test, label_train, label_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels  # stratify保持类别分布
    )

    # 训练随机森林
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # 多线程
    )
    classifier.fit(feature_train, label_train)

    return classifier, feature_test, label_test


def test_rf(
        classifier: RandomForestClassifier,
        features: np.ndarray,
        labels: np.ndarray
) -> None:
    """
    测试随机森林
    :param classifier: 训练好的随机森林
    :param features: 测试特征
    :param labels: 测试标签
    """
    prediction = classifier.predict(features)
    accuracy = accuracy_score(labels, prediction)

    print("=" * 50)
    print("随机森林测试结果:")
    print(f"准确率: {accuracy:.4f}")
    print("=" * 50)


def main():
    # 1. 数据加载与预处理
    df = pd.read_csv(
        'data/train.csv',
        nrows=1000,  # 测试用小样本，实际可去掉
        usecols=['bmi', 'waist_to_hip_ratio', 'cholesterol_total',
                 'triglycerides', 'family_history_diabetes', 'diagnosed_diabetes']
    )

    # 处理缺失值（关键！）
    df = df.fillna(df.mean())  # 用均值填充缺失值

    print("数据基本信息:")
    print(df.info())
    print("\n前5行数据:")
    print(df.head())

    # 2. 特征和标签分离
    X = df.drop('diagnosed_diabetes', axis=1).values.astype(np.float32)
    y = df['diagnosed_diabetes'].values.astype(np.float32).reshape(-1, 1)  # [样本数, 1]

    # 数据标准化（对神经网络至关重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    # 3. 训练并测试逻辑斯蒂回归
    print("\n=== 训练逻辑斯蒂回归 ===")
    lg_model = LGRegression(in_channels=X_train.shape[1])
    lg_model = train_model(
        model=lg_model,
        features=X_train_tensor,
        labels=y_train_tensor,
        loss_fn=nn.BCELoss(reduction='mean'),
        lr=0.01,
        epoch_num=EPOCH_NUM
    )
    print("\n=== 测试逻辑斯蒂回归 ===")
    test_model(lg_model, X_test_tensor, y_test_tensor)

    # 4. 训练并测试MLP
    print("\n=== 训练MLP ===")
    mlp_model = MLP(in_channels=X_train.shape[1], hidden_dim=64)
    mlp_model = train_model(
        model=mlp_model,
        features=X_train_tensor,
        labels=y_train_tensor,
        loss_fn=nn.BCELoss(reduction='mean'),
        lr=0.001,  # MLP学习率更小
        epoch_num=EPOCH_NUM
    )
    print("\n=== 测试MLP ===")
    test_model(mlp_model, X_test_tensor, y_test_tensor)

    # 5. 训练并测试随机森林（sklearn版本）
    print("\n=== 训练随机森林 ===")
    # 随机森林不需要标准化（可选），标签转回一维int
    rf_classifier, rf_X_test, rf_y_test = train_rf(
        features=X,  # 原始特征（随机森林对尺度不敏感）
        labels=df['diagnosed_diabetes'].values.astype(int)  # 一维int标签
    )
    print("\n=== 测试随机森林 ===")
    test_rf(rf_classifier, rf_X_test, rf_y_test)

    return 0


if __name__ == '__main__':
    main()
