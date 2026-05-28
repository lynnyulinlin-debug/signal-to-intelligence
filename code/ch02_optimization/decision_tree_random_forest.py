"""
实验2.5：决策树与随机森林
对应章节：第2章 - 优化算法与传统机器学习
目标：理解树模型和集成学习
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
N_TRAIN = 220
N_TEST = 1200


def generate_xor_data(n_samples, noise_std=0.85):
    x = np.random.uniform(-3.0, 3.0, size=(n_samples, 2))
    logits = x[:, 0] * x[:, 1] + 0.6 * np.sin(2.4 * x[:, 0]) - 0.45 * np.cos(2.0 * x[:, 1])
    logits += np.random.randn(n_samples) * noise_std
    y = (logits > 0).astype(int)
    flip_mask = np.random.rand(n_samples) < 0.08
    y[flip_mask] = 1 - y[flip_mask]
    return x, y


X_train, y_train = generate_xor_data(N_TRAIN)
X_test, y_test = generate_xor_data(N_TEST)

print("=" * 70)
print("Decision Tree and Random Forest: Tree Models and Ensemble Learning")
print("=" * 70)
print(f"Training samples: {N_TRAIN}")
print(f"Test samples: {N_TEST}")
print()


class SimpleDecisionTree:
    def __init__(self, max_depth=3, max_features=None, min_samples_split=2):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        feature_indices = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < X.shape[1]:
            feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 40:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 25))
                thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                gini = (
                    left_mask.sum() * self._gini(y[left_mask])
                    + right_mask.sum() * self._gini(y[right_mask])
                ) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            return {"type": "leaf", "value": np.mean(y)}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"type": "leaf", "value": np.mean(y)}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        return self._predict_sample(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class SimpleRandomForest:
    def __init__(self, n_trees=31, max_depth=6, max_features=1, min_samples_split=4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


# Make the single tree intentionally more variance-prone.
dt = SimpleDecisionTree(max_depth=12, min_samples_split=2)
dt.fit(X_train, y_train)
train_pred_dt = (dt.predict(X_train) > 0.5).astype(int)
test_pred_dt = (dt.predict(X_test) > 0.5).astype(int)
train_acc_dt = np.mean(train_pred_dt == y_train)
test_acc_dt = np.mean(test_pred_dt == y_test)

a_rf = SimpleRandomForest(n_trees=41, max_depth=6, max_features=1, min_samples_split=5)
a_rf.fit(X_train, y_train)
train_pred_rf = (a_rf.predict(X_train) > 0.5).astype(int)
test_pred_rf = (a_rf.predict(X_test) > 0.5).astype(int)
train_acc_rf = np.mean(train_pred_rf == y_train)
test_acc_rf = np.mean(test_pred_rf == y_test)

print(f"Decision tree train accuracy: {train_acc_dt:.4f}")
print(f"Decision tree test accuracy:  {test_acc_dt:.4f}")
print(f"Random forest train accuracy: {train_acc_rf:.4f}")
print(f"Random forest test accuracy:  {test_acc_rf:.4f}")
print()
print(f"Generalization gap (tree): {train_acc_dt - test_acc_dt:.4f}")
print(f"Generalization gap (forest): {train_acc_rf - test_acc_rf:.4f}")
print()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
h = 0.05
x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5, max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5, max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

plots = [
    (
        axes[0],
        dt.predict(grid).reshape(xx.shape),
        "Single Decision Tree",
        train_acc_dt,
        test_acc_dt,
        train_pred_dt,
    ),
    (
        axes[1],
        a_rf.predict(grid).reshape(xx.shape),
        "Random Forest",
        train_acc_rf,
        test_acc_rf,
        train_pred_rf,
    ),
]

for ax, surface, title, train_acc, test_acc, train_pred in plots:
    ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.45)
    ax.contour(xx, yy, surface, levels=[0.5], colors="black", linewidths=1.4)

    train_correct = train_pred == y_train
    ax.scatter(
        X_train[(y_train == 0) & train_correct, 0],
        X_train[(y_train == 0) & train_correct, 1],
        c="#1f77b4", marker="o", s=28, alpha=0.9, label="Train class 0",
    )
    ax.scatter(
        X_train[(y_train == 1) & train_correct, 0],
        X_train[(y_train == 1) & train_correct, 1],
        c="#d62728", marker="^", s=30, alpha=0.9, label="Train class 1",
    )
    ax.scatter(
        X_train[~train_correct, 0],
        X_train[~train_correct, 1],
        facecolors="none", edgecolors="black", marker="s", s=55, linewidths=1.1, label="Train mistakes",
    )
    ax.scatter(
        X_test[y_test == 0, 0],
        X_test[y_test == 0, 1],
        c="#1f77b4", marker=".", s=8, alpha=0.12,
    )
    ax.scatter(
        X_test[y_test == 1, 0],
        X_test[y_test == 1, 1],
        c="#d62728", marker=".", s=8, alpha=0.12,
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(
        f"{title}\nTrain acc={train_acc:.3f}, Test acc={test_acc:.3f}, Gap={train_acc - test_acc:.3f}"
    )
    ax.grid(True, alpha=0.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig('assets/ch02_decision_tree_random_forest.png', dpi=120, bbox_inches='tight')
print("Figure saved to: assets/ch02_decision_tree_random_forest.png")
