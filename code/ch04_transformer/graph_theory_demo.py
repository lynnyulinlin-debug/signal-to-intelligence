"""
图论基础 - 代码实验

本实验演示：
1. 图的表示（邻接矩阵、邻接表）
2. 图的遍历（DFS、BFS）
3. 注意力机制作为图的解释
4. 图的可视化
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
from collections import deque, defaultdict
import networkx as nx

# 设置随机种子
np.random.seed(42)

print("=" * 70)
print("图论基础 - 代码实验")
print("=" * 70)

# ============================================================================
# 1. 图的表示
# ============================================================================
print("\n1. 图的表示")
print("-" * 70)

# 创建一个简单的图
n_nodes = 5
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]

# 邻接矩阵表示
adjacency_matrix = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1  # 无向图

print(f"节点数: {n_nodes}")
print(f"边数: {len(edges)}")
print(f"\n邻接矩阵:")
print(adjacency_matrix.astype(int))

# 邻接表表示
adjacency_list = defaultdict(list)
for i, j in edges:
    adjacency_list[i].append(j)
    adjacency_list[j].append(i)

print(f"\n邻接表:")
for node in range(n_nodes):
    print(f"  节点 {node}: {adjacency_list[node]}")

# 度矩阵
degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
print(f"\n度矩阵:")
print(degree_matrix.astype(int))

# 拉普拉斯矩阵
laplacian_matrix = degree_matrix - adjacency_matrix
print(f"\n拉普拉斯矩阵:")
print(laplacian_matrix.astype(int))

# ============================================================================
# 2. 图的遍历
# ============================================================================
print("\n2. 图的遍历")
print("-" * 70)

def dfs(start, adjacency_list, n_nodes):
    """深度优先搜索"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # 按逆序添加邻接节点，保证正序遍历
            for neighbor in sorted(adjacency_list[node], reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order

def bfs(start, adjacency_list, n_nodes):
    """广度优先搜索"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adjacency_list[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order

dfs_order = dfs(0, adjacency_list, n_nodes)
bfs_order = bfs(0, adjacency_list, n_nodes)

print(f"DFS 遍历顺序（从节点0开始）: {dfs_order}")
print(f"BFS 遍历顺序（从节点0开始）: {bfs_order}")

# ============================================================================
# 3. 注意力机制作为图
# ============================================================================
print("\n3. 注意力机制作为图")
print("-" * 70)

# 模拟 Transformer 中的注意力
seq_len = 4
d_model = 8

# 生成查询、键、值向量
np.random.seed(42)
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

# 计算注意力权重
scores = Q @ K.T / np.sqrt(d_model)
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

print(f"序列长度: {seq_len}")
print(f"模型维度: {d_model}")
print(f"\n注意力权重矩阵（可以看作图的邻接矩阵）:")
print(attention_weights)

# 注意力权重可以看作图的边权重
print(f"\n注意力权重统计:")
for i in range(seq_len):
    print(f"  位置 {i} 的注意力分布: {attention_weights[i]}")
    max_attention_idx = np.argmax(attention_weights[i])
    print(f"    → 最关注位置 {max_attention_idx}（权重 {attention_weights[i, max_attention_idx]:.4f}）")

# ============================================================================
# 4. 图的可视化
# ============================================================================
print("\n4. 生成可视化图表")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 子图1：图的结构
ax = axes[0, 0]
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
nx.draw_networkx_edges(G, pos, width=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
ax.set_title('Graph Structure', fontsize=12, fontweight='bold')
ax.axis('off')

# 子图2：邻接矩阵热力图
ax = axes[0, 1]
im = ax.imshow(adjacency_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(n_nodes))
ax.set_yticks(range(n_nodes))
ax.set_xlabel('Node')
ax.set_ylabel('Node')
ax.set_title('Adjacency Matrix', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax)

# 子图3：度分布
ax = axes[1, 0]
degrees = [len(adjacency_list[i]) for i in range(n_nodes)]
ax.bar(range(n_nodes), degrees, color='steelblue', alpha=0.7)
ax.set_xlabel('Node')
ax.set_ylabel('Degree')
ax.set_title('Node Degree Distribution', fontsize=12, fontweight='bold')
ax.set_xticks(range(n_nodes))
ax.grid(True, alpha=0.3, axis='y')

# 子图4：注意力权重热力图
ax = axes[1, 1]
im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Attention Weights as an Adjacency Matrix', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax)

# 添加数值标签
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('assets/ch04_graph_theory.png', dpi=150, bbox_inches='tight')
print("图表已保存到 assets/ch04_graph_theory.png")

# ============================================================================
# 5. 图的性质分析
# ============================================================================
print("\n5. 图的性质分析")
print("-" * 70)

# 计算特征值
eigenvalues = np.linalg.eigvals(laplacian_matrix)
eigenvalues = np.sort(eigenvalues)

print(f"拉普拉斯矩阵的特征值: {eigenvalues}")
print(f"最小特征值: {eigenvalues[0]:.6f} (应该接近0)")
print(f"第二小特征值（代数连通性）: {eigenvalues[1]:.6f}")

# 连通性分析
print(f"\n连通性分析:")
print(f"  图是连通的: {nx.is_connected(G)}")
print(f"  连通分量数: {nx.number_connected_components(G)}")

# 路径分析
print(f"\n路径分析:")
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        try:
            path = nx.shortest_path(G, i, j)
            print(f"  节点 {i} 到节点 {j} 的最短路径: {path}（长度 {len(path)-1}）")
        except nx.NetworkXNoPath:
            print(f"  节点 {i} 到节点 {j} 无路径")

# ============================================================================
# 6. 注意力机制的图论解释
# ============================================================================
print("\n6. 注意力机制的图论解释")
print("-" * 70)

print("Transformer 中的自注意力可以看作一个完全图：")
print(f"  - 节点数: {seq_len}（序列长度）")
print(f"  - 边权重: 注意力权重 α_ij")
print(f"  - 消息传递: output_i = Σ_j α_ij * V_j")
print(f"\n这就是在图上进行的加权求和邻接节点的特征！")

# 可视化注意力图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 注意力图的可视化
ax = axes[0]
G_attention = nx.DiGraph()
for i in range(seq_len):
    for j in range(seq_len):
        if attention_weights[i, j] > 0.1:  # 只显示权重较大的边
            G_attention.add_edge(i, j, weight=attention_weights[i, j])

pos_attention = nx.circular_layout(G_attention)
nx.draw_networkx_nodes(G_attention, pos_attention, node_color='lightgreen',
                       node_size=800, ax=ax)
nx.draw_networkx_labels(G_attention, pos_attention, font_size=12,
                        font_weight='bold', ax=ax)

# 绘制边，边的宽度表示权重
edges = G_attention.edges()
weights = [G_attention[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G_attention, pos_attention, width=[w*5 for w in weights],
                       edge_color='gray', alpha=0.6, ax=ax,
                       connectionstyle='arc3,rad=0.1', arrows=True,
                       arrowsize=20, arrowstyle='->')

ax.set_title('Attention as a Directed Graph\n(Edge width indicates attention weight)',
             fontsize=12, fontweight='bold')
ax.axis('off')

# 注意力权重分布
ax = axes[1]
for i in range(seq_len):
    ax.plot(range(seq_len), attention_weights[i], marker='o', label=f'Position {i}')
ax.set_xlabel('Key Position')
ax.set_ylabel('Attention Weight')
ax.set_title('Attention Distribution for Each Query Position', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch04_attention_graph.png', dpi=150, bbox_inches='tight')
print("注意力图表已保存到 assets/ch04_attention_graph.png")

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)
