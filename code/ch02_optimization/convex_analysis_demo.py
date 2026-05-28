"""
凸分析基础 - 代码实验

本实验演示：
1. 凸函数 vs 非凸函数
2. 凸优化 vs 非凸优化
3. 梯度下降在凸和非凸问题上的表现
4. 条件数对优化的影响
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
np.random.seed(42)

print("=" * 70)
print("凸分析基础 - 代码实验")
print("=" * 70)

# ============================================================================
# 1. 凸函数 vs 非凸函数
# ============================================================================
print("\n1. 凸函数 vs 非凸函数")
print("-" * 70)

# 定义函数
def convex_function(x):
    """凸函数：f(x) = x^2"""
    return x**2

def non_convex_function(x):
    """非凸函数：f(x) = sin(x) + 0.1*x^2"""
    return np.sin(x) + 0.1 * x**2

def rosenbrock(x, y):
    """Rosenbrock 函数（非凸）：f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x)**2 + 100 * (y - x**2)**2

# 一维函数
x_curve = np.linspace(-3, 3, 1000)
y_convex = convex_function(x_curve)
y_non_convex = non_convex_function(x_curve)

print("凸函数：f(x) = x^2")
print("  - 最小值在 x=0，f(0)=0")
print("  - 任何局部最小值都是全局最小值")

print("\n非凸函数：f(x) = sin(x) + 0.1*x^2")
print("  - 有多个局部最小值")
print("  - 全局最小值不容易找到")

# ============================================================================
# 2. 梯度下降在凸和非凸问题上的表现
# ============================================================================
print("\n2. 梯度下降在凸和非凸问题上的表现")
print("-" * 70)

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=100):
    """梯度下降算法"""
    x = x0
    history = [x]
    loss_history = [f(x)]

    for _ in range(max_iter):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x)
        loss_history.append(f(x))

        # 早停
        if abs(loss_history[-1] - loss_history[-2]) < 1e-6:
            break

    return np.array(history), np.array(loss_history)

# 凸函数的梯度
def grad_convex(x):
    return 2 * x

# 非凸函数的梯度
def grad_non_convex(x):
    return np.cos(x) + 0.2 * x

# 从不同初始点开始
initial_points = [-2.5, -1.0, 0.5, 2.0]
colors = ['red', 'blue', 'green', 'orange']

print("凸函数优化（从不同初始点）:")
convex_results = []
for x0, color in zip(initial_points, colors):
    history, loss = gradient_descent(convex_function, grad_convex, x0,
                                     learning_rate=0.1, max_iter=50)
    convex_results.append((history, loss))
    final_x = history[-1]
    final_loss = loss[-1]
    print(f"  初始点 x0={x0:5.1f} → 最终 x={final_x:7.4f}, f(x)={final_loss:.6f}")

print("\n非凸函数优化（从不同初始点）:")
non_convex_results = []
for x0, color in zip(initial_points, colors):
    history, loss = gradient_descent(non_convex_function, grad_non_convex, x0,
                                     learning_rate=0.01, max_iter=100)
    non_convex_results.append((history, loss))
    final_x = history[-1]
    final_loss = loss[-1]
    print(f"  初始点 x0={x0:5.1f} → 最终 x={final_x:7.4f}, f(x)={final_loss:.6f}")

# ============================================================================
# 3. 二维 Rosenbrock 函数（非凸优化）
# ============================================================================
print("\n3. 二维 Rosenbrock 函数（非凸优化）")
print("-" * 70)

def grad_rosenbrock(x, y):
    """Rosenbrock 函数的梯度"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# 梯度下降在 Rosenbrock 函数上
def gradient_descent_2d(f, grad_f, x0, learning_rate=0.001, max_iter=1000):
    """二维梯度下降"""
    x = x0.copy()
    history = [x.copy()]
    loss_history = [f(x[0], x[1])]

    for _ in range(max_iter):
        grad = grad_f(x[0], x[1])
        x = x - learning_rate * grad
        history.append(x.copy())
        loss_history.append(f(x[0], x[1]))

        if abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break

    return np.array(history), np.array(loss_history)

# 从不同初始点开始
initial_points_2d = [
    np.array([-1.5, 2.5]),
    np.array([0.0, 0.0]),
    np.array([-0.5, -0.5])
]

print("Rosenbrock 函数优化（全局最小值在 (1, 1)）:")
rosenbrock_results = []
for i, x0 in enumerate(initial_points_2d):
    history, loss = gradient_descent_2d(rosenbrock, grad_rosenbrock, x0,
                                        learning_rate=0.001, max_iter=2000)
    rosenbrock_results.append((history, loss))
    final_x = history[-1]
    final_loss = loss[-1]
    print(f"  初始点 ({x0[0]:5.1f}, {x0[1]:5.1f}) → 最终 ({final_x[0]:7.4f}, {final_x[1]:7.4f}), f={final_loss:.6f}")

# ============================================================================
# 4. 条件数对优化的影响
# ============================================================================
print("\n4. 条件数对优化的影响")
print("-" * 70)

def quadratic_function(x, A):
    """二次函数：f(x) = 0.5 * x^T A x"""
    return 0.5 * x @ A @ x

def grad_quadratic(x, A):
    """二次函数的梯度"""
    return A @ x

# 创建不同条件数的矩阵
def create_matrix_with_condition_number(n, kappa):
    """创建条件数为 kappa 的矩阵"""
    # 特征值从 1 到 kappa
    eigenvalues = np.logspace(0, np.log10(kappa), n)
    # 随机正交矩阵
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # A = Q * diag(eigenvalues) * Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    return A

# 测试不同条件数
condition_numbers = [1, 10, 100, 1000]
n = 2
quadratic_x0 = np.ones(n)

print("二次函数优化（不同条件数）:")
for kappa in condition_numbers:
    A = create_matrix_with_condition_number(n, kappa)
    actual_kappa = np.linalg.cond(A)

    # 梯度下降
    x = quadratic_x0.copy()
    loss_history = []
    for _ in range(100):
        loss = quadratic_function(x, A)
        loss_history.append(loss)
        grad = grad_quadratic(x, A)
        x = x - 0.01 * grad

    final_loss = loss_history[-1]
    print(f"  条件数 κ={actual_kappa:8.1f} → 最终损失 {final_loss:.6e}")

# ============================================================================
# 5. 可视化
# ============================================================================
print("\n5. 生成可视化图表")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))

# 子图1：凸函数
ax = fig.add_subplot(2, 3, 1)
ax.plot(x_curve, y_convex, 'b-', linewidth=2, label='f(x) = x²')
for i, (x0, color) in enumerate(zip(initial_points, colors)):
    history, loss = convex_results[i]
    ax.plot(history, convex_function(history), 'o-', color=color,
            markersize=4, alpha=0.6, label=f'x₀={x0}')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Convex Function Optimization', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图2：非凸函数
ax = fig.add_subplot(2, 3, 2)
ax.plot(x_curve, y_non_convex, 'b-', linewidth=2, label='f(x) = sin(x) + 0.1x²')
for i, (x0, color) in enumerate(zip(initial_points, colors)):
    history, loss = non_convex_results[i]
    ax.plot(history, non_convex_function(history), 'o-', color=color,
            markersize=4, alpha=0.6, label=f'x₀={x0}')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Non-Convex Function Optimization', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图3：凸函数的损失曲线
ax = fig.add_subplot(2, 3, 3)
for i, (x0, color) in enumerate(zip(initial_points, colors)):
    history, loss = convex_results[i]
    ax.semilogy(loss, color=color, linewidth=2, label=f'x₀={x0}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Convex Function: Convergence Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 子图4：非凸函数的损失曲线
ax = fig.add_subplot(2, 3, 4)
for i, (x0, color) in enumerate(zip(initial_points, colors)):
    history, loss = non_convex_results[i]
    ax.semilogy(loss, color=color, linewidth=2, label=f'x₀={x0}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Non-Convex Function: Convergence Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 子图5：Rosenbrock 函数的等高线和优化路径
ax = fig.add_subplot(2, 3, 5)
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

levels = np.logspace(-1, 3.5, 20)
contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)

# 绘制优化路径
for i, (history, loss) in enumerate(rosenbrock_results):
    ax.plot(history[:, 0], history[:, 1], 'o-', markersize=4, alpha=0.7,
            label=f'Start {i+1}')

# 标记全局最小值
ax.plot(1, 1, 'r*', markersize=20, label='Global Minimum (1,1)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Rosenbrock Function: Optimization Path', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图6：条件数的影响
ax = fig.add_subplot(2, 3, 6)
condition_numbers_plot = []
final_losses = []

for kappa in condition_numbers:
    A = create_matrix_with_condition_number(n, kappa)
    actual_kappa = np.linalg.cond(A)

    x_condition = quadratic_x0.copy()
    for _ in range(100):
        grad = grad_quadratic(x_condition, A)
        x_condition = x_condition - 0.01 * grad

    final_loss = quadratic_function(x_condition, A)
    condition_numbers_plot.append(actual_kappa)
    final_losses.append(final_loss)

ax.loglog(condition_numbers_plot, final_losses, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Condition Number kappa(A)')
ax.set_ylabel('Final Loss (log scale)')
ax.set_title('Effect of Condition Number on Convergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('assets/ch02_convex_analysis.png', dpi=150, bbox_inches='tight')
print("图表已保存到 assets/ch02_convex_analysis.png")

# ============================================================================
# 6. 凸性判断
# ============================================================================
print("\n6. 凸性判断（通过 Hessian 矩阵）")
print("-" * 70)

def hessian_convex_2d(x, y):
    """f(x,y) = x^2 + y^2 的 Hessian"""
    return np.array([[2, 0], [0, 2]])

def hessian_non_convex_2d(x, y):
    """f(x,y) = x^2 - y^2 的 Hessian（鞍点）"""
    return np.array([[2, 0], [0, -2]])

def hessian_rosenbrock(x, y):
    """Rosenbrock 函数的 Hessian"""
    return np.array([
        [2 - 400*y + 1200*x**2, -400*x],
        [-400*x, 200]
    ])

# 检查凸性
H_convex = hessian_convex_2d(0, 0)
H_non_convex = hessian_non_convex_2d(0, 0)
H_rosenbrock = hessian_rosenbrock(0, 0)

print("凸函数 f(x,y) = x² + y²:")
eigenvalues_convex = np.linalg.eigvals(H_convex)
print(f"  Hessian 特征值: {eigenvalues_convex}")
print(f"  是否半正定: {np.all(eigenvalues_convex >= -1e-10)}")

print("\n非凸函数 f(x,y) = x² - y²（鞍点）:")
eigenvalues_non_convex = np.linalg.eigvals(H_non_convex)
print(f"  Hessian 特征值: {eigenvalues_non_convex}")
print(f"  是否半正定: {np.all(eigenvalues_non_convex >= -1e-10)}")

print("\nRosenbrock 函数在 (0,0):")
eigenvalues_rosenbrock = np.linalg.eigvals(H_rosenbrock)
print(f"  Hessian 特征值: {eigenvalues_rosenbrock}")
print(f"  是否半正定: {np.all(eigenvalues_rosenbrock >= -1e-10)}")

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)
