# Local Deep Researcher

Local Deep Researcher is a fully local web research assistant that uses any LLM hosted by [Ollama](https://ollama.com/search) or [LMStudio](https://lmstudio.ai/). Give it a topic and it will generate a web search query, gather web search results, summarize the results of web search, reflect on the summary to examine knowledge gaps, generate a new search query to address the gaps, and repeat for a user-defined number of cycles. It will provide the user a final markdown summary with all sources used to generate the summary.

![ollama-deep-research](https://github.com/user-attachments/assets/1c6b28f8-6b64-42ba-a491-1ab2875d50ea)

Short summary video:
<video src="https://github.com/user-attachments/assets/02084902-f067-4658-9683-ff312cab7944" controls></video>

## 🔥 Updates 

* 8/6/25: Added support for tool calling and [gpt-oss](https://openai.com/index/introducing-gpt-oss/). 

> ⚠️ **WARNING (8/6/25)**: The `gpt-oss` models do not support JSON mode in Ollama. Select `use_tool_calling` in the configuration to use tool calling instead of JSON mode.

## 📺 Video Tutorials

See it in action or build it yourself? Check out these helpful video tutorials:

- [Overview of Local Deep Researcher with R1](https://www.youtube.com/watch?v=sGUjmyfof4Q) - Load and test [DeepSeek R1](https://api-docs.deepseek.com/news/news250120) [distilled models](https://ollama.com/library/deepseek-r1).
- [Building Local Deep Researcher from Scratch](https://www.youtube.com/watch?v=XGuTzHoqlj8) - Overview of how this is built.

## 🚀 Quickstart

Clone the repository:

```shell
git clone https://github.com/langchain-ai/local-deep-researcher.git
cd local-deep-researcher
```

Then edit the `.env` file to customize the environment variables according to your needs. These environment variables control the model selection, search tools, and other configuration settings. When you run the application, these values will be automatically loaded via `python-dotenv` (because `langgraph.json` point to the "env" file).

```shell
cp .env.example .env
```

### Selecting local model with Ollama

1. Download the Ollama app for Mac [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b):

   ```shell
   ollama pull deepseek-r1:8b
   ```

3. Optionally, update the `.env` file with the following Ollama configuration settings.

- If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`.

```shell
LLM_PROVIDER=ollama
OLLAMA_BASE_URL="http://localhost:11434" # Ollama service endpoint, defaults to `http://localhost:11434`
LOCAL_LLM=model # the model to use, defaults to `llama3.2` if not set
```

### Selecting local model with LMStudio

1. Download and install LMStudio from [here](https://lmstudio.ai/).

2. In LMStudio:

   - Download and load your preferred model (e.g., qwen_qwq-32b)
   - Go to the "Local Server" tab
   - Start the server with the OpenAI-compatible API
   - Note the server URL (default: <http://localhost:1234/v1>)

3. Optionally, update the `.env` file with the following LMStudio configuration settings.

- If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`.

```shell
LLM_PROVIDER=lmstudio
LOCAL_LLM=qwen_qwq-32b  # Use the exact model name as shown in LMStudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

### Selecting search tool

By default, it will use [DuckDuckGo](https://duckduckgo.com/) for web search, which does not require an API key. But you can also use [SearXNG](https://docs.searxng.org/), [Tavily](https://tavily.com/) or [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) by adding their API keys to the environment file. Optionally, update the `.env` file with the following search tool configuration and API keys. If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`.

```shell
SEARCH_API=xxx # the search API to use, such as `duckduckgo` (default)
TAVILY_API_KEY=xxx # the tavily API key to use
PERPLEXITY_API_KEY=xxx # the perplexity API key to use
MAX_WEB_RESEARCH_LOOPS=xxx # the maximum number of research loop steps, defaults to `3`
FETCH_FULL_PAGE=xxx # fetch the full page content (with `duckduckgo`), defaults to `false`
```

### Running with LangGraph Studio

#### Mac

1. (Recommended) Create a virtual environment:

   ```bash python -m venv .venv
   source .venv/bin/activate
   ```

2. Launch LangGraph server:

   ```bash
   # Install uv package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
   ```

#### Windows

1. (Recommended) Create a virtual environment:

   - Install `Python 3.11` (and add to PATH during installation).
   - Restart your terminal to ensure Python is available, then create and activate a virtual environment:

     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Launch LangGraph server:

   ```powershell
   # Install dependencies
   pip install -e .
   pip install -U "langgraph-cli[inmem]"

   # Start the LangGraph server
   langgraph dev
   ```

### Using the LangGraph Studio UI

When you launch LangGraph server, you should see the following output and Studio will open in your browser:

> Ready!  
> API: <http://127.0.0.1:2024>  
> Docs: <http://127.0.0.1:2024/docs>  
> LangGraph Studio Web UI: <https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>

Open `LangGraph Studio Web UI` via the URL above. In the `configuration` tab, you can directly set various assistant configurations. Keep in mind that the priority order for configuration values is:

```txt
1. Environment variables (highest priority)
2. LangGraph UI configuration
3. Default values in the Configuration class (lowest priority)
```

![Screenshot 2025-01-24 at 10 08 31 PM](https://github.com/user-attachments/assets/7cfd0e04-28fd-4cfa-aee5-9a556d74ab21)

Give the assistant a topic for research, and you can visualize its process!

![Screenshot 2025-01-24 at 10 08 22 PM](https://github.com/user-attachments/assets/4de6bd89-4f3b-424c-a9cb-70ebd3d45c5f)

### Model Compatibility Note

When selecting a local LLM, set steps use structured JSON output. Some models may have difficulty with this requirement, and the assistant has fallback mechanisms to handle this. As an example, the [DeepSeek R1 (7B)](https://ollama.com/library/deepseek-llm:7b) and [DeepSeek R1 (1.5B)](https://ollama.com/library/deepseek-r1:1.5b) models have difficulty producing required JSON output, and the assistant will use a fallback mechanism to handle this.

### Browser Compatibility Note

When accessing the LangGraph Studio UI:

- Firefox is recommended for the best experience
- Safari users may encounter security warnings due to mixed content (HTTPS/HTTP)
- If you encounter issues, try:
  1. Using Firefox or another browser
  2. Disabling ad-blocking extensions
  3. Checking browser console for specific error messages

## How it works

Local Deep Researcher is inspired by [IterDRAG](https://arxiv.org/html/2410.04343v1#:~:text=To%20tackle%20this%20issue%2C%20we,used%20to%20generate%20intermediate%20answers.). This approach will decompose a query into sub-queries, retrieve documents for each one, answer the sub-query, and then build on the answer by retrieving docs for the second sub-query. Here, we do similar:

- Given a user-provided topic, use a local LLM (via [Ollama](https://ollama.com/search) or [LMStudio](https://lmstudio.ai/)) to generate a web search query
- Uses a search engine / tool to find relevant sources
- Uses LLM to summarize the findings from web search related to the user-provided research topic
- Then, it uses the LLM to reflect on the summary, identifying knowledge gaps
- It generates a new search query to address the knowledge gaps
- The process repeats, with the summary being iteratively updated with new information from web search
- Runs for a configurable number of iterations (see `configuration` tab)

## Outputs

The output of the graph is a markdown file containing the research summary, with citations to the sources used. All sources gathered during research are saved to the graph state. You can visualize them in the graph state, which is visible in LangGraph Studio:

![Screenshot 2024-12-05 at 4 08 59 PM](https://github.com/user-attachments/assets/e8ac1c0b-9acb-4a75-8c15-4e677e92f6cb)

The final summary is saved to the graph state as well:

![Screenshot 2024-12-05 at 4 10 11 PM](https://github.com/user-attachments/assets/f6d997d5-9de5-495f-8556-7d3891f6bc96)

## Deployment Options

There are [various ways](https://langchain-ai.github.io/langgraph/concepts/#deployment-options) to deploy this graph. See [Module 6](https://github.com/langchain-ai/langchain-academy/tree/main/module-6) of LangChain Academy for a detailed walkthrough of deployment options with LangGraph.

## TypeScript Implementation

A TypeScript port of this project (without Perplexity search) is available at:
<https://github.com/PacoVK/ollama-deep-researcher-ts>

## Running as a Docker container

The included `Dockerfile` only runs LangChain Studio with local-deep-researcher as a service, but does not include Ollama as a dependant service. You must run Ollama separately and configure the `OLLAMA_BASE_URL` environment variable. Optionally you can also specify the Ollama model to use by providing the `LOCAL_LLM` environment variable.

Clone the repo and build an image:

```sh
docker build -t local-deep-researcher .
```

Run the container:

```sh
$ docker run --rm -it -p 2024:2024 \
  -e SEARCH_API="tavily" \
  -e TAVILY_API_KEY="tvly-***YOUR_KEY_HERE***" \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434/" \
  -e LOCAL_LLM="llama3.2" \
  local-deep-researcher
```

NOTE: You will see log message:

```sh
2025-02-10T13:45:04.784915Z [info     ] 🎨 Opening Studio in your browser... [browser_opener] api_variant=local_dev message=🎨 Opening Studio in your browser...
URL: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
```

...but the browser will not launch from the container.

Instead, visit this link with the correct baseUrl IP address: [`https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024)

## Running with PostgreSQL Persistence via Docker Compose

The `langgraph dev` method described previously uses in-memory storage, which means the state of your research runs (conversation history, intermediate steps) is lost when the server stops.

To enable **persistent storage**—allowing you to resume runs, inspect past states, and ensure durability across restarts—you can use the provided Docker Compose setup which includes a PostgreSQL database for checkpointing.

This method uses the `langgraph.json` configuration file to generate a specific `Dockerfile` tailored for this persistent setup, then builds and runs the application using Docker Compose.

**Steps:**

1. **(Prerequisite)** A `langgraph.json` file (see `langgraph-postgres.json`) that correctly specifies dependencies (including `langgraph-checkpoint-postgres`, `psycopg2-binary`, and your graph's needs) and points to your graph definition.
2. **(Prerequisite)** Make sure your `.env` file is configured with necessary API keys (like `LANGSMITH_API_KEY`) and settings (like `LLM_PROVIDER`, `OLLAMA_BASE_URL` or `LLMSTUDIO_BASE_URL`, and `LOCAL_LLM`).
3. **Generate the Dockerfile:** Run the following command to generate the Dockerfile based on `langgraph-postgres.json`:

   ```bash
   langgraph dockerfile > postgres.Dockerfile
   ```

   _Note: In this repository, the autogenerated `postgres.Dockerfile` is `.gitignore`'d to avoid committing it to git by mistake_

4. **Run Docker Compose:** Build and start the services defined in `compose.yaml`:

   ```bash
   docker compose up --build -d
   ```

   - `--build`: Builds the `langgraph-api` image using the generated `postgres.Dockerfile`.
   - `-d`: Runs the services in detached mode (in the background).

5. **Access LangGraph Studio UI:** Once the services are up (check with `docker compose ps`), open your browser and navigate to [`https://smith.langchain.com/studio/?baseUrl=http://localhost:8100`](https://smith.langchain.com/studio/?baseUrl=http://localhost:8100), pointing it to the port exposed in `compose.yaml` (port `8100` in the example):

With this setup, the state of your research runs will be saved in the `langgraph-postgres` container's database volume. You can stop (`docker compose down`) and restart (`docker compose up -d`) the services, and the state of previously run threads will be loaded from PostgreSQL, allowing you to resume or inspect them via the LangGraph Studio UI. If you update `langgraph-postgres.json`, simply regenerate the `postgres.Dockerfile` before running `docker compose up --build -d` again.
