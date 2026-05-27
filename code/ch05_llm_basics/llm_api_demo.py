"""
第5章代码实验：LLM API调用演示

本实验演示如何调用LLM API，包括：
1. 基本的API调用
2. Prompt工程
3. In-Context Learning
"""

import os
from typing import Optional

# 注意：需要设置环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY
# export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"


def demo_basic_api_call():
    """演示基本的API调用"""
    print("=" * 50)
    print("演示1：基本的API调用")
    print("=" * 50)
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic()
        
        # 简单的问题
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "什么是机器学习？用一句话解释。"}
            ]
        )
        
        print("问题：什么是机器学习？用一句话解释。")
        print(f"回答：{message.content[0].text}")
        print()
        
    except ImportError:
        print("需要安装 anthropic 库：pip install anthropic")
    except Exception as e:
        print(f"API调用失败：{e}")
        print("请确保设置了 ANTHROPIC_API_KEY 环境变量")


def demo_prompt_engineering():
    """演示Prompt工程"""
    print("=" * 50)
    print("演示2：Prompt工程")
    print("=" * 50)
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic()
        
        # 不好的Prompt
        print("不好的Prompt：")
        print("问题：翻译这个")
        print()
        
        # 好的Prompt
        print("好的Prompt：")
        prompt = """你是一个专业的翻译。
        
将以下英文翻译成中文，保持原意：

"The quick brown fox jumps over the lazy dog"

请只返回翻译结果，不需要其他说明。"""
        
        print(f"问题：{prompt}")
        print()
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print(f"回答：{message.content[0].text}")
        print()
        
    except ImportError:
        print("需要安装 anthropic 库：pip install anthropic")
    except Exception as e:
        print(f"API调用失败：{e}")


def demo_in_context_learning():
    """演示In-Context Learning"""
    print("=" * 50)
    print("演示3：In-Context Learning")
    print("=" * 50)
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic()
        
        # Few-shot学习：通过示例让模型学习
        prompt = """你是一个情感分类器。

示例：
- "这个产品很好用，我很满意。" → 正面
- "太差了，完全不值这个价格。" → 负面
- "一般般，没什么特别的。" → 中立

现在分类以下句子：
"这个服务很快，客服也很友好。"

请只返回分类结果（正面/负面/中立），不需要其他说明。"""
        
        print(f"问题：{prompt}")
        print()
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print(f"回答：{message.content[0].text}")
        print()
        
    except ImportError:
        print("需要安装 anthropic 库：pip install anthropic")
    except Exception as e:
        print(f"API调用失败：{e}")


def demo_multi_turn_conversation():
    """演示多轮对话"""
    print("=" * 50)
    print("演示4：多轮对话")
    print("=" * 50)
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic()
        
        # 多轮对话
        messages = [
            {"role": "user", "content": "什么是Transformer？"},
        ]
        
        print("用户：什么是Transformer？")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )
        
        assistant_message = response.content[0].text
        print(f"助手：{assistant_message}")
        print()
        
        # 继续对话
        messages.append({"role": "assistant", "content": assistant_message})
        messages.append({"role": "user", "content": "它与RNN有什么区别？"})
        
        print("用户：它与RNN有什么区别？")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )
        
        print(f"助手：{response.content[0].text}")
        print()
        
    except ImportError:
        print("需要安装 anthropic 库：pip install anthropic")
    except Exception as e:
        print(f"API调用失败：{e}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("第5章：LLM基础 - API调用演示")
    print("=" * 50 + "\n")
    
    # 运行演示
    demo_basic_api_call()
    demo_prompt_engineering()
    demo_in_context_learning()
    demo_multi_turn_conversation()
    
    print("=" * 50)
    print("演示完成！")
    print("=" * 50)
