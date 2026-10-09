# Local Deep Researcher：本地深度研究助手

Local Deep Researcher 是一个完全在本地运行的网页研究助手，可以使用由 [Ollama](https://ollama.com/search) 或 [LMStudio](https://lmstudio.ai/) 托管的大语言模型（LLM）。输入一个研究主题后，它会生成搜索查询、收集网页搜索结果并进行总结，再反思总结中尚未解决的问题，生成新的查询以补充缺失的信息。上述过程会按照用户设定的轮数重复执行，最终输出一份 Markdown 总结，并列出生成总结时使用的全部来源。

![本地深度研究助手工作流程](https://github.com/user-attachments/assets/1c6b28f8-6b64-42ba-a491-1ab2875d50ea)

简短演示视频：
<video src="https://github.com/user-attachments/assets/02084902-f067-4658-9683-ff312cab7944" controls></video>

## 🔥 更新记录

* 8/6/25：新增对工具调用和 [gpt-oss](https://openai.com/index/introducing-gpt-oss/) 的支持。

> ⚠️ **注意（8/6/25）**：`gpt-oss` 模型在 Ollama 中不支持 JSON 模式。请在配置中启用 `use_tool_calling`，使用工具调用替代 JSON 模式。

## 📺 视频教程

希望查看实际运行效果，或者自己动手搭建？可以参考以下视频教程：
- [使用 R1 的 Local Deep Researcher 概览](https://www.youtube.com/watch?v=sGUjmyfof4Q)：加载并测试 [DeepSeek R1](https://api-docs.deepseek.com/news/news250120) 的[蒸馏模型](https://ollama.com/library/deepseek-r1)。
- [从零构建 Local Deep Researcher](https://www.youtube.com/watch?v=XGuTzHoqlj8)：介绍项目的构建过程。

## 🚀 快速开始

克隆仓库：
```shell
git clone https://github.com/langchain-ai/local-deep-researcher.git
cd local-deep-researcher
```

随后按需编辑 `.env` 文件，设置环境变量。这些变量用于指定模型、搜索工具及其他运行配置。运行应用时，`python-dotenv` 会自动加载这些值，因为 `langgraph.json` 指定了相应的环境变量文件。
```shell
cp .env.example .env
```

### 使用 Ollama 选择本地模型

1. 从[下载页面](https://ollama.com/download)下载 Mac 版 Ollama 应用。

2. 从 [Ollama](https://ollama.com/search) 拉取一个本地大语言模型。例如，可以使用[这个模型](https://ollama.com/library/deepseek-r1:8b)：
```shell
ollama pull deepseek-r1:8b
```

3. 根据需要，在 `.env` 文件中更新以下 Ollama 配置。

* 显式设置的值会覆盖 `configuration.py` 中 `Configuration` 类的默认值。
```shell
LLM_PROVIDER=ollama
OLLAMA_BASE_URL="http://localhost:11434" # Ollama 服务地址，默认值为 http://localhost:11434
LOCAL_LLM=model # 使用的模型名称；未设置时默认为 llama3.2
```

### 使用 LMStudio 选择本地模型

1. 从[官方网站](https://lmstudio.ai/)下载并安装 LMStudio。

2. 在 LMStudio 中：
   - 下载并加载你选择的模型，例如 `qwen_qwq-32b`。
   - 打开“Local Server”（本地服务器）选项卡。
   - 启动提供 OpenAI 兼容 API 的服务器。
   - 记下服务器地址，默认值为 `http://localhost:1234/v1`。

3. 根据需要，在 `.env` 文件中更新以下 LMStudio 配置。

* 显式设置的值会覆盖 `configuration.py` 中 `Configuration` 类的默认值。
```shell
LLM_PROVIDER=lmstudio
LOCAL_LLM=qwen_qwq-32b  # 请使用 LMStudio 中显示的完整模型名称
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

### 选择搜索工具

默认使用 [DuckDuckGo](https://duckduckgo.com/) 进行网页搜索，无需 API Key。也可以使用 [SearXNG](https://docs.searxng.org/)、[Tavily](https://tavily.com/) 或 [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api)，并在环境变量文件中填写相应服务需要的 API Key。根据需要，在 `.env` 文件中更新以下搜索配置；显式设置的值会覆盖 `configuration.py` 中 `Configuration` 类的默认值。
```shell
SEARCH_API=xxx # 使用的搜索接口，例如默认的 duckduckgo
TAVILY_API_KEY=xxx # 使用的 Tavily API Key
PERPLEXITY_API_KEY=xxx # 使用的 Perplexity API Key
MAX_WEB_RESEARCH_LOOPS=xxx # 网页研究的最大循环次数，默认为 3
FETCH_FULL_PAGE=xxx # 是否抓取网页全文（使用 duckduckgo 时），默认为 false
```

### 通过 LangGraph Studio 运行

#### Mac

1. 创建虚拟环境（推荐）：
```bash
python -m venv .venv
source .venv/bin/activate
```

2. 启动 LangGraph 服务器：

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Windows

1. 创建虚拟环境（推荐）：

* 安装 `Python 3.11`，并在安装时将其添加到 PATH。
* 重新打开终端，确认 Python 可用，然后创建并激活虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. 启动 LangGraph 服务器：

```powershell
# 安装依赖
pip install -e .
pip install -U "langgraph-cli[inmem]"

# 启动 LangGraph 服务器
langgraph dev
```

### 使用 LangGraph Studio 界面

启动 LangGraph 服务器后，终端应显示以下输出，Studio 会在浏览器中打开：
> Ready!

> API: http://127.0.0.1:2024

> Docs: http://127.0.0.1:2024/docs

> LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

通过上面的地址打开 LangGraph Studio 网页界面。在 `configuration` 选项卡中，可以直接调整助手的配置。配置值的优先级如下：

```text
1. 环境变量（优先级最高）
2. LangGraph 界面配置
3. Configuration 类中的默认值（优先级最低）
```

<img width="1621" alt="配置界面截图：2025-01-24 22:08:31" src="https://github.com/user-attachments/assets/7cfd0e04-28fd-4cfa-aee5-9a556d74ab21" />

输入研究主题后，就可以在界面中查看助手的执行过程。

<img width="1621" alt="研究流程截图：2025-01-24 22:08:22" src="https://github.com/user-attachments/assets/4de6bd89-4f3b-424c-a9cb-70ebd3d45c5f" />

### 模型兼容性说明

选择本地大语言模型时，请注意：部分步骤需要结构化的 JSON 输出。有些模型难以满足这一要求，助手会通过回退机制处理。例如，[DeepSeek R1（7B）](https://ollama.com/library/deepseek-llm:7b) 和 [DeepSeek R1（1.5B）](https://ollama.com/library/deepseek-r1:1.5b) 可能难以生成所需的 JSON，遇到这种情况时会使用回退机制。

### 浏览器兼容性说明

访问 LangGraph Studio 界面时：
- 推荐使用 Firefox，以获得更好的使用体验。
- Safari 用户可能会遇到由 HTTPS/HTTP 混合内容引起的安全提示。
- 遇到问题时，可以尝试：
  1. 使用 Firefox 或其他浏览器。
  2. 关闭广告拦截扩展。
  3. 查看浏览器控制台中的具体错误信息。

## 工作原理

Local Deep Researcher 受到 [IterDRAG](https://arxiv.org/html/2410.04343v1#:~:text=To%20tackle%20this%20issue%2C%20we,used%20to%20generate%20intermediate%20answers.) 的启发。该方法会将查询拆分为子查询，分别检索文档、回答子问题，并在已有回答的基础上继续检索后续子问题所需的资料。本项目采用类似流程：
- 根据用户提供的主题，通过 [Ollama](https://ollama.com/search) 或 [LMStudio](https://lmstudio.ai/) 中的本地大语言模型生成网页搜索查询。
- 使用搜索引擎或搜索工具查找相关来源。
- 使用大语言模型总结搜索结果中与研究主题相关的信息。
- 让模型反思当前总结，识别缺失或尚待澄清的信息。
- 为这些问题生成新的搜索查询。
- 重复上述过程，利用新检索到的信息逐步更新总结。
- 按配置的迭代次数运行，具体设置见 `configuration` 选项卡。

## 输出结果

图的输出是一份 Markdown 文件，包含研究总结及所用来源的引用。研究过程中收集到的全部来源都会保存在图状态中，可以通过 LangGraph Studio 查看：

![来源状态截图：2024-12-05 16:08:59](https://github.com/user-attachments/assets/e8ac1c0b-9acb-4a75-8c15-4e677e92f6cb)

最终总结也会保存在图状态中：

![最终总结截图：2024-12-05 16:10:11](https://github.com/user-attachments/assets/f6d997d5-9de5-495f-8556-7d3891f6bc96)

## 部署方式

可以通过[多种方式](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)部署该图。有关 LangGraph 部署方案的详细教程，请参考 LangChain Academy 的[模块 6](https://github.com/langchain-ai/langchain-academy/tree/main/module-6)。

## TypeScript 实现

本项目的 TypeScript 移植版本可在以下仓库获取，该版本不包含 Perplexity 搜索：
https://github.com/PacoVK/ollama-deep-researcher-ts

## 通过 Docker 容器运行

项目提供的 `Dockerfile` 仅启动以 local-deep-researcher 为服务的 LangChain Studio，不包含 Ollama 服务。你需要单独运行 Ollama，并设置 `OLLAMA_BASE_URL` 环境变量。也可以通过 `LOCAL_LLM` 环境变量指定要使用的 Ollama 模型。

克隆仓库后，构建镜像：
```shell
$ docker build -t local-deep-researcher .
```

运行容器：

> 排版说明：下面仅清除了原命令中续行反斜杠后的多余空格，命令参数保持不变。

```shell
$ docker run --rm -it -p 2024:2024 \
  -e SEARCH_API="tavily" \
  -e TAVILY_API_KEY="tvly-***YOUR_KEY_HERE***" \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434/" \
  -e LOCAL_LLM="llama3.2" \
  local-deep-researcher
```

注意：日志中会出现类似下面的信息，以下保留程序输出的原文：
```text
2025-02-10T13:45:04.784915Z [info     ] 🎨 Opening Studio in your browser... [browser_opener] api_variant=local_dev message=🎨 Opening Studio in your browser...
URL: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
```

但是，容器不会自动打开宿主机的浏览器。

请在浏览器中访问已使用正确 `baseUrl` IP 地址的链接：[`https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024)。
