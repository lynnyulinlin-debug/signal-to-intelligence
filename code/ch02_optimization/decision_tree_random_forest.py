"""
实验2.5：决策树与随机森林
对应章节：第2章 - 优化算法与传统机器学习
目标：理解树模型和集成学习
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
N_SAMPLES = 200

# ============ 生成数据 ============
# 生成二分类数据
X = np.random.randn(N_SAMPLES, 2) * 2
y = ((X[:, 0] > 0) & (X[:, 1] > 0)) | ((X[:, 0] < 0) & (X[:, 1] < 0))
y = y.astype(int)

print("=" * 70)
print("决策树与随机森林：树模型和集成学习")
print("=" * 70)

# ============ 简化的决策树 ============
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        """计算基尼系数"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        """找到最佳分割"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                gini = (len(y[left_mask]) * self._gini(y[left_mask]) +
                       len(y[right_mask]) * self._gini(y[right_mask])) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建树"""
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'type': 'leaf', 'value': np.mean(y)}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return {'type': 'leaf', 'value': np.mean(y)}

        left_mask = X[:, feature] <= threshold

        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        """预测单个样本"""
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

# 训练决策树
dt = SimpleDecisionTree(max_depth=4)
dt.fit(X, y)
y_pred_dt = (dt.predict(X) > 0.5).astype(int)
accuracy_dt = np.mean(y_pred_dt == y)

print(f"决策树准确率: {accuracy_dt:.4f}")

# ============ 随机森林 ============
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap采样
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # 训练树
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# 训练随机森林
rf = SimpleRandomForest(n_trees=10, max_depth=4)
rf.fit(X, y)
y_pred_rf = (rf.predict(X) > 0.5).astype(int)
accuracy_rf = np.mean(y_pred_rf == y)

print(f"随机森林准确率: {accuracy_rf:.4f}")
print()

# ============ 可视化 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 决策树决策边界
ax = axes[0]
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z_dt = (dt.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5).astype(int)
Z_dt = Z_dt.reshape(xx.shape)

ax.contourf(xx, yy, Z_dt, levels=1, colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', s=30, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='x', s=30, label='Class 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title(f'Decision Tree (Accuracy: {accuracy_dt:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 随机森林决策边界
ax = axes[1]
Z_rf = (rf.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5).astype(int)
Z_rf = Z_rf.reshape(xx.shape)

ax.contourf(xx, yy, Z_rf, levels=1, colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', s=30, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='x', s=30, label='Class 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title(f'Random Forest (Accuracy: {accuracy_rf:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch02_decision_tree_random_forest.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch02_decision_tree_random_forest.png")
