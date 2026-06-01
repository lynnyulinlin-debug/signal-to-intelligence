"""
第6章代码实验：Agent ReAct 循环演示

演示 ReAct（Reason + Act + Observe）模式：
- LLM 先推理（Reason）决定调用哪个工具
- 执行工具（Act）获取结果
- 观察结果（Observe）决定是否继续
- 重复直到得出答案或达到最大步数

使用模拟数据，无需 API Key 或外部依赖。
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "assets/ch06_agent_error_accumulation.png"


# ── 模拟工具库 ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "transformer": "Transformer 是基于自注意力机制的神经网络架构，由 Vaswani 等人于 2017 年提出。",
    "rag": "RAG（检索增强生成）先从知识库检索相关文档，再用文档增强 LLM 的生成过程。",
    "lora": "LoRA 通过低秩分解降低微调成本，只训练约 0.1-1% 的参数即可达到接近全量微调的效果。",
    "scaling": "Chinchilla 缩放律表明：模型参数量和训练数据量应同步增长，最优比例约为 20 tokens/参数。",
}


def search_knowledge(query: str) -> str:
    """在知识库中搜索相关信息（模拟语义检索）"""
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    return "未找到相关信息。"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})  # 限制 eval 作用域
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


TOOLS = {
    "search": {
        "description": "在知识库中搜索信息。适用于：需要查找概念定义、技术原理。",
        "func": search_knowledge,
        "param": "query（搜索关键词）",
    },
    "calculate": {
        "description": "计算数学表达式。适用于：数值计算、参数量估算。",
        "func": calculate,
        "param": "expression（Python 数学表达式）",
    },
}


# ── 模拟 LLM 推理（规则替代，展示 ReAct 结构）────────────────────────────────

def simulate_llm_reason(question: str, history: list[dict]) -> dict:
    """
    模拟 LLM 的推理步骤，返回：
      {"action": "search"|"calculate"|"answer", "input": ..., "thought": ...}
    真实场景中这里是 LLM API 调用。
    """
    step = len(history)

    # 场景1：单步搜索问题
    if "什么是" in question or "介绍" in question:
        if step == 0:
            keyword = question.replace("什么是", "").replace("？", "").replace("?", "").strip()
            return {
                "thought": f"用户想了解 '{keyword}' 的定义，需要搜索知识库。",
                "action": "search",
                "input": keyword,
            }
        else:
            obs = history[-1]["observation"]
            return {
                "thought": f"已获得搜索结果，可以直接回答。",
                "action": "answer",
                "input": obs,
            }

    # 场景2：需要计算的问题
    if "参数量" in question or "计算" in question:
        if step == 0:
            return {
                "thought": "用户问参数量，需要先搜索 LoRA 的参数比例，再计算具体数值。",
                "action": "search",
                "input": "lora",
            }
        elif step == 1:
            return {
                "thought": "已知 LoRA 训练约 1% 参数。假设模型 70B 参数，计算 LoRA 参数量。",
                "action": "calculate",
                "input": "70_000_000_000 * 0.01",
            }
        else:
            calc_result = history[-1]["observation"]
            return {
                "thought": f"计算完成：{calc_result} 个参数，约 7 亿。可以回答了。",
                "action": "answer",
                "input": f"70B 模型使用 LoRA（r=8）微调时，可训练参数约为 {int(float(calc_result)):,} 个（约 7 亿），是全量微调的 1%。",
            }

    # 默认：直接回答
    return {
        "thought": "问题较简单，无需工具，直接回答。",
        "action": "answer",
        "input": "这个问题超出了当前知识库范围。",
    }


# ── ReAct 主循环 ─────────────────────────────────────────────────────────────

def react_agent(question: str, max_steps: int = 5) -> str:
    """
    ReAct 循环：Reason → Act → Observe → Reason → ...
    返回最终答案。
    """
    print(f"\n{'='*55}")
    print(f"问题：{question}")
    print(f"{'='*55}")

    history = []

    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}]")

        # Reason：LLM 推理，决定下一步行动
        decision = simulate_llm_reason(question, history)
        print(f"  Thought : {decision['thought']}")
        print(f"  Action  : {decision['action']}")

        if decision["action"] == "answer":
            print(f"  Answer  : {decision['input']}")
            return decision["input"]

        # Act：执行工具
        tool_name = decision["action"]
        tool_input = decision["input"]
        print(f"  Input   : {tool_input}")

        tool_func = TOOLS[tool_name]["func"]
        observation = tool_func(tool_input)

        # Observe：记录观察结果
        print(f"  Observe : {observation}")
        history.append({
            "step": step,
            "action": tool_name,
            "input": tool_input,
            "observation": observation,
        })

    return "达到最大步数，未能得出答案。"


# ── 演示错误累积效应 ──────────────────────────────────────────────────────────

def demo_error_accumulation():
    """演示多步推理中错误累积的风险"""
    print(f"\n{'='*55}")
    print("错误累积演示：每步成功率不同时，多步任务的整体成功率")
    print(f"{'='*55}")

    rates = [0.80, 0.90, 0.95]
    steps = list(range(1, 11))

    for r in rates:
        print(f"\n  单步成功率 {r:.0%}：")
        for n in [1, 3, 5, 10]:
            print(f"    {n:2d} 步任务：{r**n:.1%} 成功率")

    print("\n结论：Agent 步骤越多，整体可靠性越低。")
    print("工程实践：控制 Agent 步数，复杂任务拆分为多个简单 Agent。")

    # ── 生成图表 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    labels = ["80% per step", "90% per step", "95% per step"]

    for r, color, label in zip(rates, colors, labels):
        overall = [r ** n for n in steps]
        ax.plot(steps, [v * 100 for v in overall],
                marker="o", markersize=5, color=color, label=label, linewidth=2)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(10.1, 50, "50%", va="center", fontsize=9, color="gray")

    ax.set_xlabel("Number of Steps", fontsize=12)
    ax.set_ylabel("Overall Success Rate (%)", fontsize=12)
    ax.set_title("Agent Error Accumulation: Multi-Step Reliability", fontsize=13, fontweight="bold")
    ax.set_xticks(steps)
    ax.set_ylim(0, 105)
    ax.legend(title="Per-step success rate", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存：{OUTPUT_PATH}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("第6章：Agent ReAct 循环演示")
    print("（使用模拟数据，无需 API Key）\n")

    # 场景1：单步工具调用
    react_agent("什么是 RAG？")

    # 场景2：多步工具调用（搜索 + 计算）
    react_agent("70B 模型用 LoRA 微调需要多少参数量？")

    # 错误累积演示
    demo_error_accumulation()

    print(f"\n{'='*55}")
    print("演示完成！")
    print(f"{'='*55}")
