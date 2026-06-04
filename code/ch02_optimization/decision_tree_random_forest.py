"""
实验2.5：决策树与随机森林
对应章节：第2章 - 优化算法与传统机器学习
目标：理解树模型和集成学习
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_TRAIN = 220
N_TEST = 1200
OUTPUT_PATH = Path("assets/ch02_decision_tree_random_forest.png")


def generate_xor_data(n_samples, noise_std=0.85, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    x = rng.uniform(-3.0, 3.0, size=(n_samples, 2))
    logits = x[:, 0] * x[:, 1] + 0.6 * np.sin(2.4 * x[:, 0]) - 0.45 * np.cos(2.0 * x[:, 1])
    logits += rng.randn(n_samples) * noise_std
    y = (logits > 0).astype(int)
    flip_mask = rng.rand(n_samples) < 0.08
    y[flip_mask] = 1 - y[flip_mask]
    return x, y


def bootstrap_sample(x, y=None, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    indices = rng.choice(len(x), len(x), replace=True)
    if y is None:
        return x[indices], indices
    return x[indices], y[indices], indices


class SimpleDecisionTree:
    def __init__(self, max_depth=3, max_features=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.tree = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _best_split(self, x, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        feature_indices = np.arange(x.shape[1])
        if self.max_features is not None and self.max_features < x.shape[1]:
            feature_indices = self.rng.choice(x.shape[1], self.max_features, replace=False)

        for feature in feature_indices:
            thresholds = np.unique(x[:, feature])
            if len(thresholds) > 40:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 25))
                thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_split
                    or right_mask.sum() < self.min_samples_split
                ):
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

    def _build_tree(self, x, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return {"type": "leaf", "value": np.mean(y)}

        feature, threshold = self._best_split(x, y)
        if feature is None:
            return {"type": "leaf", "value": np.mean(y)}

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(x[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(x[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, x, y):
        self.tree = self._build_tree(x, y)
        return self

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        return self._predict_sample(x, node["right"])

    def predict(self, x):
        return np.array([self._predict_sample(item, self.tree) for item in x])


class SimpleRandomForest:
    def __init__(
        self,
        n_trees=31,
        max_depth=6,
        max_features=1,
        min_samples_split=4,
        random_state=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for tree_index in range(self.n_trees):
            indices = self.rng.choice(len(x), len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=tree_index,
            )
            tree.fit(x_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        return np.mean(predictions, axis=0)


def run_experiment(seed=42):
    x_train, y_train = generate_xor_data(N_TRAIN, seed=seed)
    x_test, y_test = generate_xor_data(N_TEST, seed=seed + 1)

    dt = SimpleDecisionTree(max_depth=12, min_samples_split=2, random_state=seed)
    dt.fit(x_train, y_train)
    train_pred_dt = (dt.predict(x_train) > 0.5).astype(int)
    test_pred_dt = (dt.predict(x_test) > 0.5).astype(int)
    train_acc_dt = np.mean(train_pred_dt == y_train)
    test_acc_dt = np.mean(test_pred_dt == y_test)

    rf = SimpleRandomForest(
        n_trees=41,
        max_depth=6,
        max_features=1,
        min_samples_split=5,
        random_state=seed,
    )
    rf.fit(x_train, y_train)
    train_pred_rf = (rf.predict(x_train) > 0.5).astype(int)
    test_pred_rf = (rf.predict(x_test) > 0.5).astype(int)
    train_acc_rf = np.mean(train_pred_rf == y_train)
    test_acc_rf = np.mean(test_pred_rf == y_test)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "dt": dt,
        "rf": rf,
        "train_pred_dt": train_pred_dt,
        "train_pred_rf": train_pred_rf,
        "train_acc_dt": train_acc_dt,
        "test_acc_dt": test_acc_dt,
        "train_acc_rf": train_acc_rf,
        "test_acc_rf": test_acc_rf,
    }


def print_summary(result):
    print("=" * 70)
    print("Decision Tree and Random Forest: Tree Models and Ensemble Learning")
    print("=" * 70)
    print(f"Training samples: {len(result['X_train'])}")
    print(f"Test samples: {len(result['X_test'])}")
    print()
    print(f"Decision tree train accuracy: {result['train_acc_dt']:.4f}")
    print(f"Decision tree test accuracy:  {result['test_acc_dt']:.4f}")
    print(f"Random forest train accuracy: {result['train_acc_rf']:.4f}")
    print(f"Random forest test accuracy:  {result['test_acc_rf']:.4f}")
    print()
    print(f"Generalization gap (tree): {result['train_acc_dt'] - result['test_acc_dt']:.4f}")
    print(f"Generalization gap (forest): {result['train_acc_rf'] - result['test_acc_rf']:.4f}")
    print()


def plot_results(result, output_path=OUTPUT_PATH):
    x_train = result["X_train"]
    y_train = result["y_train"]
    x_test = result["X_test"]
    y_test = result["y_test"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    h = 0.05
    x_min = min(x_train[:, 0].min(), x_test[:, 0].min()) - 0.5
    x_max = max(x_train[:, 0].max(), x_test[:, 0].max()) + 0.5
    y_min = min(x_train[:, 1].min(), x_test[:, 1].min()) - 0.5
    y_max = max(x_train[:, 1].max(), x_test[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    plots = [
        (
            axes[0],
            result["dt"].predict(grid).reshape(xx.shape),
            "Single Decision Tree",
            result["train_acc_dt"],
            result["test_acc_dt"],
            result["train_pred_dt"],
        ),
        (
            axes[1],
            result["rf"].predict(grid).reshape(xx.shape),
            "Random Forest",
            result["train_acc_rf"],
            result["test_acc_rf"],
            result["train_pred_rf"],
        ),
    ]

    for ax, surface, title, train_acc, test_acc, train_pred in plots:
        ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.45)
        ax.contour(xx, yy, surface, levels=[0.5], colors="black", linewidths=1.4)

        train_correct = train_pred == y_train
        ax.scatter(
            x_train[(y_train == 0) & train_correct, 0],
            x_train[(y_train == 0) & train_correct, 1],
            c="#1f77b4",
            marker="o",
            s=28,
            alpha=0.9,
            label="Train class 0",
        )
        ax.scatter(
            x_train[(y_train == 1) & train_correct, 0],
            x_train[(y_train == 1) & train_correct, 1],
            c="#d62728",
            marker="^",
            s=30,
            alpha=0.9,
            label="Train class 1",
        )
        ax.scatter(
            x_train[~train_correct, 0],
            x_train[~train_correct, 1],
            facecolors="none",
            edgecolors="black",
            marker="s",
            s=55,
            linewidths=1.1,
            label="Train mistakes",
        )
        ax.scatter(
            x_test[y_test == 0, 0],
            x_test[y_test == 0, 1],
            c="#1f77b4",
            marker=".",
            s=8,
            alpha=0.12,
        )
        ax.scatter(
            x_test[y_test == 1, 0],
            x_test[y_test == 1, 1],
            c="#d62728",
            marker=".",
            s=8,
            alpha=0.12,
        )

        gap = train_acc - test_acc
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"{title}\nTrain acc={train_acc:.3f}, Test acc={test_acc:.3f}, Gap={gap:.3f}")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
