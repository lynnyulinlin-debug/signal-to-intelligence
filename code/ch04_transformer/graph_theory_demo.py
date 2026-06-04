"""
图论基础 - 代码实验

本实验演示：
1. 图的表示（邻接矩阵、邻接表）
2. 图的遍历（DFS、BFS）
3. 注意力机制作为图的解释
4. 图的可视化
"""

from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch04_graph_theory.png")
ATTENTION_PATH = Path("assets/ch04_attention_graph.png")


def build_sample_graph():
    n_nodes = 5
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]

    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    adjacency_list = defaultdict(list)
    for i, j in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    return {
        "n_nodes": n_nodes,
        "edges": edges,
        "adjacency_matrix": adjacency_matrix,
        "adjacency_list": adjacency_list,
        "degree_matrix": degree_matrix,
        "laplacian_matrix": laplacian_matrix,
    }


def dfs(start, adjacency_list):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in sorted(adjacency_list[node], reverse=True):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def bfs(start, adjacency_list):
    visited = {start}
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adjacency_list[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def build_attention_demo(seed=42, seq_len=4, d_model=8):
    rng = np.random.RandomState(seed)
    q = rng.randn(seq_len, d_model)
    k = rng.randn(seq_len, d_model)
    v = rng.randn(seq_len, d_model)

    scores = q @ k.T / np.sqrt(d_model)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    attention_weights = np.exp(shifted)
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "attention_weights": attention_weights,
    }


def run_experiment(seed=42):
    graph = build_sample_graph()
    attention = build_attention_demo(seed=seed)

    dfs_order = dfs(0, graph["adjacency_list"])
    bfs_order = bfs(0, graph["adjacency_list"])

    eigenvalues = np.sort(np.linalg.eigvals(graph["laplacian_matrix"]))
    G = nx.Graph()
    G.add_edges_from(graph["edges"])

    shortest_paths = {}
    for i in range(graph["n_nodes"]):
        for j in range(i + 1, graph["n_nodes"]):
            shortest_paths[(i, j)] = nx.shortest_path(G, i, j)

    G_attention = nx.DiGraph()
    for i in range(attention["seq_len"]):
        for j in range(attention["seq_len"]):
            if attention["attention_weights"][i, j] > 0.1:
                G_attention.add_edge(i, j, weight=float(attention["attention_weights"][i, j]))

    return {
        **graph,
        **attention,
        "dfs_order": dfs_order,
        "bfs_order": bfs_order,
        "eigenvalues": eigenvalues,
        "is_connected": nx.is_connected(G),
        "connected_components": nx.number_connected_components(G),
        "shortest_paths": shortest_paths,
        "attention_graph": G_attention,
    }


def print_summary(result):
    print("=" * 70)
    print("图论基础 - 代码实验")
    print("=" * 70)

    print("\n1. 图的表示")
    print("-" * 70)
    print(f"节点数: {result['n_nodes']}")
    print(f"边数: {len(result['edges'])}")
    print("\n邻接矩阵:")
    print(result["adjacency_matrix"].astype(int))
    print("\n邻接表:")
    for node in range(result["n_nodes"]):
        print(f"  节点 {node}: {result['adjacency_list'][node]}")
    print("\n度矩阵:")
    print(result["degree_matrix"].astype(int))
    print("\n拉普拉斯矩阵:")
    print(result["laplacian_matrix"].astype(int))

    print("\n2. 图的遍历")
    print("-" * 70)
    print(f"DFS 遍历顺序（从节点0开始）: {result['dfs_order']}")
    print(f"BFS 遍历顺序（从节点0开始）: {result['bfs_order']}")

    print("\n3. 注意力机制作为图")
    print("-" * 70)
    print(f"序列长度: {result['seq_len']}")
    print(f"模型维度: {result['d_model']}")
    print("\n注意力权重矩阵（可以看作图的邻接矩阵）:")
    print(result["attention_weights"])
    print("\n注意力权重统计:")
    for i in range(result["seq_len"]):
        max_attention_idx = int(np.argmax(result["attention_weights"][i]))
        print(f"  位置 {i} 的注意力分布: {result['attention_weights'][i]}")
        print(f"    → 最关注位置 {max_attention_idx}")
        print(f"      权重 {result['attention_weights'][i, max_attention_idx]:.4f}")

    print("\n5. 图的性质分析")
    print("-" * 70)
    print(f"拉普拉斯矩阵的特征值: {result['eigenvalues']}")
    print(f"最小特征值: {result['eigenvalues'][0]:.6f} (应该接近0)")
    print(f"第二小特征值（代数连通性）: {result['eigenvalues'][1]:.6f}")
    print("\n连通性分析:")
    print(f"  图是连通的: {result['is_connected']}")
    print(f"  连通分量数: {result['connected_components']}")
    print("\n路径分析:")
    for (i, j), path in result["shortest_paths"].items():
        print(f"  节点 {i} 到节点 {j} 的最短路径: {path}（长度 {len(path) - 1}）")

    print("\n6. 注意力机制的图论解释")
    print("-" * 70)
    print("Transformer 中的自注意力可以看作一个完全图：")
    print(f"  - 节点数: {result['seq_len']}（序列长度）")
    print("  - 边权重: 注意力权重 α_ij")
    print("  - 消息传递: output_i = Σ_j α_ij * V_j")
    print("\n这就是在图上进行的加权求和邻接节点的特征！")
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH, attention_output_path=ATTENTION_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    graph = nx.Graph()
    graph.add_edges_from(result["edges"])
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=500, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=2, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold", ax=ax)
    ax.set_title("Graph Structure", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = axes[0, 1]
    im = ax.imshow(result["adjacency_matrix"], cmap="Blues", aspect="auto")
    ax.set_xticks(range(result["n_nodes"]))
    ax.set_yticks(range(result["n_nodes"]))
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    ax.set_title("Adjacency Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    degrees = [len(result["adjacency_list"][i]) for i in range(result["n_nodes"])]
    ax.bar(range(result["n_nodes"]), degrees, color="steelblue", alpha=0.7)
    ax.set_xlabel("Node")
    ax.set_ylabel("Degree")
    ax.set_title("Node Degree Distribution", fontsize=12, fontweight="bold")
    ax.set_xticks(range(result["n_nodes"]))
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    im = ax.imshow(result["attention_weights"], cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(result["seq_len"]))
    ax.set_yticks(range(result["seq_len"]))
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("Attention Weights as an Adjacency Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(result["seq_len"]):
        for j in range(result["seq_len"]):
            ax.text(
                j,
                i,
                f"{result['attention_weights'][i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    g_attention = result["attention_graph"]
    if g_attention.number_of_edges() == 0:
        for i in range(result["seq_len"]):
            g_attention.add_edge(i, i, weight=float(result["attention_weights"][i, i]))

    pos_attention = nx.circular_layout(g_attention)
    nx.draw_networkx_nodes(
        g_attention,
        pos_attention,
        node_color="lightgreen",
        node_size=800,
        ax=ax[0],
    )
    nx.draw_networkx_labels(g_attention, pos_attention, font_size=12, font_weight="bold", ax=ax[0])
    edges = g_attention.edges()
    weights = [g_attention[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        g_attention,
        pos_attention,
        width=[w * 5 for w in weights],
        edge_color="gray",
        alpha=0.6,
        ax=ax[0],
        connectionstyle="arc3,rad=0.1",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
    )
    ax[0].set_title(
        "Attention as a Directed Graph\n(Edge width indicates attention weight)",
        fontsize=12,
        fontweight="bold",
    )
    ax[0].axis("off")

    for i in range(result["seq_len"]):
        ax[1].plot(
            range(result["seq_len"]),
            result["attention_weights"][i],
            marker="o",
            label=f"Position {i}",
        )
    ax[1].set_xlabel("Key Position")
    ax[1].set_ylabel("Attention Weight")
    ax[1].set_title(
        "Attention Distribution for Each Query Position",
        fontsize=12,
        fontweight="bold",
    )
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    attention_output_path = Path(attention_output_path)
    attention_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(attention_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, attention_output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path, attention_output_path = plot_results(result)
    print(f"图表已保存到 {output_path}")
    print(f"注意力图表已保存到 {attention_output_path}")


if __name__ == "__main__":
    main()
