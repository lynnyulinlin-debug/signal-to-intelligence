"""
第6章代码实验：RAG（检索增强生成）演示

本实验演示如何构建一个简单的RAG系统，包括：
1. 构建知识库
2. 检索相关文档
3. 生成增强的回答
"""

from typing import List, Dict
import json


class SimpleRAG:
    """简单的RAG系统实现"""
    
    def __init__(self):
        """初始化RAG系统"""
        self.knowledge_base = [
            {
                "id": 1,
                "title": "什么是Transformer",
                "content": "Transformer是一种基于自注意力机制的神经网络架构。它由编码器和解码器组成，能够并行处理序列数据。"
            },
            {
                "id": 2,
                "title": "什么是LLM",
                "content": "大语言模型（LLM）是用大规模文本数据预训练的深度学习模型。它能够理解和生成自然语言。"
            },
            {
                "id": 3,
                "title": "什么是RAG",
                "content": "检索增强生成（RAG）是一种结合检索和生成的方法。它先从知识库检索相关文档，然后用这些文档增强生成过程。"
            },
            {
                "id": 4,
                "title": "什么是微调",
                "content": "微调是在预训练模型的基础上，用特定任务的数据进行训练。这样可以让模型适应特定的任务。"
            },
        ]
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """检索相关文档"""
        # 简单的关键词匹配
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.knowledge_base:
            doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
            # 计算Jaccard相似度
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            score = intersection / union if union > 0 else 0
            scores.append((doc, score))
        
        # 按相似度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k个文档
        return [doc for doc, score in scores[:top_k]]
    
    def generate_with_context(self, query: str, retrieved_docs: List[Dict]) -> str:
        """用检索到的文档生成回答"""
        # 构建增强的提示
        context = "\n".join([f"- {doc['title']}: {doc['content']}" for doc in retrieved_docs])
        
        prompt = f"""基于以下信息回答问题：

信息：
{context}

问题：{query}

回答："""
        
        return prompt
    
    def query(self, question: str) -> Dict:
        """完整的RAG流程"""
        # 1. 检索
        retrieved_docs = self.retrieve(question)
        
        # 2. 生成增强的提示
        enhanced_prompt = self.generate_with_context(question, retrieved_docs)
        
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "enhanced_prompt": enhanced_prompt,
            "num_retrieved": len(retrieved_docs)
        }


def demo_rag():
    """演示RAG系统"""
    print("=" * 50)
    print("第6章：LLM应用与微调 - RAG演示")
    print("=" * 50)
    print()
    
    # 初始化RAG系统
    rag = SimpleRAG()
    
    # 测试查询
    test_queries = [
        "什么是Transformer？",
        "如何微调模型？",
        "RAG有什么优势？"
    ]
    
    for query in test_queries:
        print(f"问题：{query}")
        print()
        
        result = rag.query(query)
        
        print(f"检索到 {result['num_retrieved']} 个相关文档：")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            print(f"  {i}. {doc['title']}")
        print()
        
        print("增强的提示：")
        print(result['enhanced_prompt'])
        print()
        print("-" * 50)
        print()


def demo_agent_framework():
    """演示Agent框架的概念"""
    print("=" * 50)
    print("Agent框架演示")
    print("=" * 50)
    print()
    
    # 定义工具
    tools = [
        {
            "name": "search",
            "description": "搜索知识库",
            "parameters": {
                "query": "搜索关键词"
            }
        },
        {
            "name": "calculate",
            "description": "进行数学计算",
            "parameters": {
                "expression": "数学表达式"
            }
        },
        {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {}
        }
    ]
    
    print("可用的工具：")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
    print()
    
    # Agent的工作流程
    print("Agent的工作流程：")
    print("1. 用户提问：'2024年有多少天？'")
    print("2. Agent分析：需要知道2024年是否是闰年")
    print("3. Agent调用工具：get_time() 获取当前年份")
    print("4. Agent调用工具：calculate(366 if leap_year else 365)")
    print("5. Agent生成回答：'2024年是闰年，有366天'")
    print()


def demo_finetuning_vs_prompt():
    """演示微调 vs Prompt工程的对比"""
    print("=" * 50)
    print("微调 vs Prompt工程对比")
    print("=" * 50)
    print()
    
    comparison = {
        "方面": ["成本", "时间", "性能", "灵活性", "数据需求"],
        "Prompt工程": ["低", "快（秒级）", "中等", "高", "无"],
        "微调": ["高", "慢（小时级）", "高", "低", "需要数据集"]
    }
    
    # 打印对比表
    print(f"{comparison['方面'][0]:<15} {comparison['Prompt工程'][0]:<15} {comparison['微调'][0]:<15}")
    print("-" * 45)
    
    for i in range(len(comparison['方面'])):
        print(f"{comparison['方面'][i]:<15} {comparison['Prompt工程'][i]:<15} {comparison['微调'][i]:<15}")
    print()
    
    print("选择建议：")
    print("- 简单任务 → 使用Prompt工程")
    print("- 复杂任务 + 有数据 → 使用微调")
    print("- 需要快速迭代 → 使用Prompt工程")
    print("- 需要最优性能 → 使用微调")
    print()


if __name__ == "__main__":
    demo_rag()
    demo_agent_framework()
    demo_finetuning_vs_prompt()
    
    print("=" * 50)
    print("演示完成！")
    print("=" * 50)
