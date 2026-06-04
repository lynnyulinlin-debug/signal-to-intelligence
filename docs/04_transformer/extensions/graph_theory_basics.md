# 4.0 图论基础

**核心问题：** 什么是图？为什么注意力机制可以看作图？

---

## 为什么需要理解图论

Transformer 中的注意力机制本质上是一个**图结构**：
- 每个 token 是一个**节点**
- token 之间的注意力权重是**边的权重**
- 注意力计算就是在图上进行的**消息传递**

理解图论可以帮助我们更深入地理解 Transformer 的工作原理。

---

## 图的基本概念

### 图的定义

一个图 $G = (V, E)$ 由以下部分组成：

- **顶点集合** $V = \{v_1, v_2, \ldots, v_n\}$：图中的节点
- **边集合** $E = \{(v_i, v_j), \ldots\}$：连接节点的边

### 图的类型

| 类型 | 定义 | 例子 |
|------|------|------|
| 无向图 | 边没有方向 | 社交网络（朋友关系） |
| 有向图 | 边有方向 | 网页链接（A→B） |
| 加权图 | 边有权重 | 注意力权重 |
| 完全图 | 任意两个节点都相连 | Transformer 中的自注意力 |

### 图的表示

#### 1. 邻接矩阵（Adjacency Matrix）

对于 $n$ 个节点的图，邻接矩阵 $\mathbf{A}$ 是 $n \times n$ 矩阵：

$$A_{ij} = \begin{cases} 
w_{ij} & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

其中 $w_{ij}$ 是边的权重。

**例子：**
```
图：1 → 2 → 3
      ↓   ↑
      └───┘

邻接矩阵：
    1  2  3
1 [ 0  1  0 ]
2 [ 0  0  1 ]
3 [ 1  0  0 ]
```

#### 2. 邻接表（Adjacency List）

对于每个节点，存储它的邻接节点列表。

**优势：** 稀疏图中节省空间

```python
# 邻接表表示
graph = {
    1: [2],
    2: [3],
    3: [1]
}
```

### 度矩阵（Degree Matrix）

对于无向图，节点 $i$ 的**度**是与它相连的边数：

$$d_i = \sum_j A_{ij}$$

**度矩阵** $\mathbf{D}$ 是对角矩阵：

$$D_{ii} = d_i, \quad D_{ij} = 0 \text{ (for } i \neq j\text{)}$$

### 拉普拉斯矩阵（Laplacian Matrix）

**定义：**
$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

**性质：**
- 对称矩阵
- 半正定矩阵
- 最小特征值为 0，对应的特征向量是全 1 向量

**应用：** 图的谱分析、图聚类、图卷积神经网络

---

## 图的遍历

### 深度优先搜索（DFS）

从一个节点开始，尽可能深地探索图。

```python
def dfs(node, graph, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited)
```

### 广度优先搜索（BFS）

从一个节点开始，按层级探索图。

```python
from collections import deque

def bfs(start, graph):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

---

## 图在神经网络中的应用

### 1. 计算图（Computational Graph）

深度学习中的反向传播可以看作在计算图上进行的消息传递：

```
输入 x
  ↓
线性层：y = Wx + b
  ↓
激活函数：z = ReLU(y)
  ↓
输出 z
```

每个操作是一个节点，数据流是边。

### 2. 图神经网络（GNN）

在图上进行神经网络计算。每个节点的表示通过聚合邻接节点的信息更新：

$$h_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \sum_{j \in N(i)} h_j^{(l)}\right)$$

其中 $N(i)$ 是节点 $i$ 的邻接节点集合。

### 3. 注意力机制作为图

**Transformer 中的自注意力可以看作一个完全图：**

- **节点**：序列中的每个 token
- **边权重**：注意力权重 $\alpha_{ij}$
- **消息传递**：每个 token 聚合所有其他 token 的信息

**注意力权重的计算：**
$$\alpha_{ij} = \frac{\exp(Q_i K_j^T / \sqrt{d_k})}{\sum_k \exp(Q_i K_k^T / \sqrt{d_k})}$$

这就是在图上计算**归一化的边权重**。

**值的聚合：**
$$\text{output}_i = \sum_j \alpha_{ij} V_j$$

这就是**加权求和邻接节点的特征**。

---

## 图的性质

### 连通性

- **连通图**：任意两个节点都有路径相连
- **强连通图**（有向图）：任意两个节点都有双向路径

### 路径和距离

- **路径**：从一个节点到另一个节点的边序列
- **最短路径**：边数最少的路径
- **距离**：最短路径的长度

**应用：** 在 Transformer 中，距离可以用来衡量 token 之间的"远近"。

### 环（Cycle）

- **无环图**：不包含任何环
- **有向无环图（DAG）**：常用于表示计算流程

---

## 图的算法

### 最短路径（Dijkstra 算法）

找到两个节点之间的最短路径。

```python
import heapq

def dijkstra(start, graph):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current_dist > distances[current]:
            continue
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

### 拓扑排序

对有向无环图进行排序，使得每条边都从前面的节点指向后面的节点。

```python
def topological_sort(graph):
    visited = set()
    stack = []
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return stack[::-1]
```

---

## 与 Transformer 的连接

### 自注意力作为图

```
Query (Q)：查询向量
Key (K)：键向量
Value (V)：值向量

注意力权重 = softmax(Q @ K^T / √d_k)
            ↑
        这是图的邻接矩阵！

输出 = 注意力权重 @ V
     ↑
   这是图上的消息传递！
```

### 多头注意力作为多个图

每个注意力头学习一个不同的图结构，捕捉不同类型的关系。

### 位置编码与图距离

位置编码可以看作是在图上编码节点之间的相对距离。

---

## 实践建议

### 何时使用图表示

- 数据有**关系结构**（如社交网络、分子结构）
- 需要**消息传递**（如 GNN）
- 需要理解**依赖关系**（如计算图）

### 常见图算法的复杂度

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| DFS/BFS | $O(V + E)$ | $O(V)$ |
| Dijkstra | $O((V + E)\log V)$ | $O(V)$ |
| 拓扑排序 | $O(V + E)$ | $O(V)$ |
| 矩阵乘法 | $O(V^3)$ | $O(V^2)$ |

---

## 关键要点

1. **图是一种通用的数据结构**，可以表示任何有关系的数据
2. **邻接矩阵**是图的数学表示，便于计算
3. **注意力机制本质上是在图上进行的消息传递**
4. **图神经网络**通过聚合邻接节点的信息来更新节点表示
5. **理解图论有助于理解 Transformer 和 GNN 的工作原理**

---

## 代码实验

- **代码文件：** [`code/ch04_transformer/graph_theory_demo.py`](../../../code/ch04_transformer/graph_theory_demo.py)
- **运行方式：** `python code/ch04_transformer/graph_theory_demo.py`

![Graph Theory](/assets/ch04_graph_theory.png)

*图E4.1：图的基本结构示意。展示节点、边、邻接矩阵与图的遍历方式。*

![Attention as Graph](/assets/ch04_attention_graph.png)

*图E4.2：注意力机制的图结构视角。每个 token 是节点，注意力权重是边权重，注意力计算即图上的消息传递。*

---

## 与后续章节的连接

- **4.1-4.4**：注意力机制作为图的应用
- **第5-8章（LLM）**：计算图和消息传递
- **扩展**：图神经网络（GNN）、知识图谱
