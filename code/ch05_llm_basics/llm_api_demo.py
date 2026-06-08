"""
第5章代码实验：LLM API 调用演示

本实验演示如何调用 LLM API，包括：
1. 基本的 API 调用
2. Prompt 工程
3. In-Context Learning
4. 多轮对话

支持的 API 提供商（按国内可用性排序）：
  - 本地 OpenAI 兼容服务（llama.cpp / Ollama）：启动本地服务后配置 LOCAL_LLM_BASE_URL
  - DeepSeek：export DEEPSEEK_API_KEY="your-key"
  - 阿里云百炼（Qwen）：export DASHSCOPE_API_KEY="your-key"
  - 智谱 AI（GLM）：export ZHIPUAI_API_KEY="your-key"
  - Anthropic（Claude）：export ANTHROPIC_API_KEY="your-key"
  - OpenAI（GPT）：export OPENAI_API_KEY="your-key"

离线优先：优先检测本地 OpenAI 兼容服务（llama.cpp / Ollama；无需 API Key）。
国内用户推荐 DeepSeek 或阿里云百炼，无需境外网络。
"""

import os

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


def get_client():
    """
    按优先级自动选择可用的 API 提供商。
    优先使用本地 OpenAI 兼容服务（llama.cpp / Ollama；离线），然后是国内 API，最后是境外 API。
    """
    if not _OPENAI_AVAILABLE:
        print("提示：openai 库未安装，跳过 OpenAI 兼容格式的提供商。")
        print("安装：pip install openai")

    # 0. 本地 OpenAI 兼容服务（llama.cpp / Ollama）
    local_base = os.environ.get("LOCAL_LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    local_model = os.environ.get("LOCAL_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
    if _OPENAI_AVAILABLE:
        try:
            local_client = OpenAI(api_key="local", base_url=local_base)
            local_client.models.list()
            print(f"使用本地 OpenAI 兼容服务（{local_model}）")
            return local_client, local_model
        except Exception:
            pass  # 本地服务未运行，继续检查远程 API

    # 1. DeepSeek（国内首选，兼容 OpenAI 格式）
    if _OPENAI_AVAILABLE and os.environ.get("DEEPSEEK_API_KEY"):
        print("使用 DeepSeek API")
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        ), "deepseek-chat"

    # 2. 阿里云百炼 / Qwen（国内备选，兼容 OpenAI 格式）
    if _OPENAI_AVAILABLE and os.environ.get("DASHSCOPE_API_KEY"):
        print("使用阿里云百炼 API（Qwen）")
        return OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ), "qwen-plus"

    # 3. 智谱 AI / GLM（兼容 OpenAI 格式）
    if _OPENAI_AVAILABLE and os.environ.get("ZHIPUAI_API_KEY"):
        print("使用智谱 AI API（GLM）")
        return OpenAI(
            api_key=os.environ["ZHIPUAI_API_KEY"],
            base_url="https://open.bigmodel.cn/api/paas/v4"
        ), "glm-4-flash"

    # 4. OpenAI（需境外网络）
    if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        print("使用 OpenAI API")
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        ), "gpt-4o-mini"

    # 5. Anthropic（需境外网络，单独处理）
    if os.environ.get("ANTHROPIC_API_KEY"):
        return None, "anthropic"

    return None, None


def call_api(client, model, messages, max_tokens=512):
    """统一的 API 调用接口，兼容 OpenAI 格式和 Anthropic 格式。"""
    if model == "anthropic":
        try:
            from anthropic import Anthropic
            ac = Anthropic()
            resp = ac.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=messages
            )
            return resp.content[0].text
        except ImportError:
            raise RuntimeError("需要安装 anthropic 库：pip install anthropic")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages
    )
    return resp.choices[0].message.content


def demo_basic_api_call(client, model):
    """演示基本的 API 调用。"""
    print("=" * 50)
    print("演示1：基本的 API 调用")
    print("=" * 50)

    messages = [{"role": "user", "content": "什么是机器学习？用一句话解释。"}]
    print("问题：什么是机器学习？用一句话解释。")
    answer = call_api(client, model, messages)
    print(f"回答：{answer}\n")


def demo_prompt_engineering(client, model):
    """演示 Prompt 工程：对比模糊 Prompt 和清晰 Prompt 的差异。"""
    print("=" * 50)
    print("演示2：Prompt 工程")
    print("=" * 50)

    # 好的 Prompt：角色 + 任务 + 约束
    prompt = """你是一个专业的翻译。

将以下英文翻译成中文，保持原意：

"The quick brown fox jumps over the lazy dog"

请只返回翻译结果，不需要其他说明。"""

    print(f"Prompt：{prompt}\n")
    answer = call_api(client, model, [{"role": "user", "content": prompt}])
    print(f"回答：{answer}\n")


def demo_in_context_learning(client, model):
    """演示 In-Context Learning（Few-shot）。"""
    print("=" * 50)
    print("演示3：In-Context Learning（Few-shot）")
    print("=" * 50)

    prompt = """你是一个情感分类器。

示例：
- "这个产品很好用，我很满意。" → 正面
- "太差了，完全不值这个价格。" → 负面
- "一般般，没什么特别的。" → 中立

现在分类以下句子：
"这个服务很快，客服也很友好。"

请只返回分类结果（正面/负面/中立），不需要其他说明。"""

    print(f"Prompt（含 3 个示例）：\n{prompt}\n")
    answer = call_api(client, model, [{"role": "user", "content": prompt}])
    print(f"回答：{answer}\n")


def demo_multi_turn_conversation(client, model):
    """演示多轮对话。"""
    print("=" * 50)
    print("演示4：多轮对话")
    print("=" * 50)

    messages = [{"role": "user", "content": "什么是 Transformer？"}]
    print("用户：什么是 Transformer？")
    reply1 = call_api(client, model, messages)
    print(f"助手：{reply1}\n")

    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "它与 RNN 有什么区别？"})
    print("用户：它与 RNN 有什么区别？")
    reply2 = call_api(client, model, messages)
    print(f"助手：{reply2}\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("第5章：LLM 基础 - API 调用演示")
    print("=" * 50 + "\n")

    client, model = get_client()

    if model is None:
        print("未找到可用的 LLM。请选择以下任意一种方式：")
        print("  离线（推荐）：")
        print("    llama.cpp / Ollama 本地服务           # OpenAI 兼容接口，无需 API Key")
        print("    export LOCAL_LLM_BASE_URL=...         # 本地服务地址（可选）")
        print("    export LOCAL_LLM_MODEL=qwen2.5:7b     # 自定义模型（可选）")
        print("  国内 API：")
        print("    export DEEPSEEK_API_KEY='your-key'    # DeepSeek")
        print("    export DASHSCOPE_API_KEY='your-key'   # 阿里云百炼")
        print("    export ZHIPUAI_API_KEY='your-key'     # 智谱 AI")
        print("  境外 API：")
        print("    export ANTHROPIC_API_KEY='your-key'   # Claude")
        print("    export OPENAI_API_KEY='your-key'      # GPT")
        print("\n完整说明见 docs/appendix/B_environment_setup.md")
        exit(1)

    print(f"模型：{model}\n")

    try:
        demo_basic_api_call(client, model)
        demo_prompt_engineering(client, model)
        demo_in_context_learning(client, model)
        demo_multi_turn_conversation(client, model)
    except Exception as e:
        print(f"\nAPI 调用失败：{e}")
        print("请检查 API Key 是否正确，以及网络是否可用。")
        exit(1)

    print("=" * 50)
    print("演示完成！")
    print("=" * 50)
