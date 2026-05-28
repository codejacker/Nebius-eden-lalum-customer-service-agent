# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers natural-language questions about the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — 26,872 labelled customer-service exchanges across 11 categories and 27 intents.

Features: multi-step tool use, persistent conversation memory (SQLite), per-user profile tracking, a query router, and a FastMCP server that exposes the dataset tools over the MCP protocol.

---

## Quick Start (≤ 5 minutes)

### 1. Clone and enter the repo

```bash
git clone https://github.com/codejacker/Nebius-eden-lalum-customer-service-agent
cd Nebius-eden-lalum-customer-service-agent
```

### 2. Set up the environment

**Mac / Linux:**
```bash
bash setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

Both scripts create a virtual environment, install dependencies, and prompt you to configure `.env`.

### 3. Add your API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
NEBIUS_API_KEY=your_nebius_key_here
LANGSMITH_API_KEY=your_langsmith_key_here   # optional — enables tracing
LANGCHAIN_TRACING_V2=true                   # optional
```

Get a Nebius key at [nebius.com](https://nebius.com). Get a LangSmith key at [smith.langchain.com](https://smith.langchain.com).

### 4. Download the dataset

```bash
python download_data.py
```

Downloads `bitext.csv` (~19 MB) to `data/`.

### 5. Run the agent

```bash
python main.py
```

Use `--session <name>` to enable persistent profile tracking across restarts:

```bash
python main.py --session alice
```

You're ready. Ask something like:

```
You: How many refund-related rows are in the dataset?
You: Show me 5 examples of customers wanting to cancel an order.
You: Summarize how agents handle account deletion requests.
```

---

## CLI Reference

```bash
# Named session — conversation and profile persisted to disk
python main.py --session alice

# Anonymous session — no profile tracking (conversation still persists)
python main.py

# RAM-only — nothing written to disk
python main.py --session alice --no-persist

# Show a session's learned profile and exit
python main.py --session alice --show-profile

# List all sessions with saved profiles
python main.py --list-sessions
```

The CLI streams reasoning in real time:

```
You: Show me 3 examples of delivery complaints.

  [TOOL] get_examples({'category': 'DELIVERY', 'n': 3})
  [OBS]  [{"category": "DELIVERY", "intent": "delivery_options", ...}]

Agent: Here are 3 delivery examples from the dataset: ...
```

---

## MCP Server

The FastMCP server exposes 5 dataset tools over the MCP protocol so any MCP-compatible client (Claude Desktop, Cursor, your own agent) can call them independently.

### Start the server

```bash
python mcp_server/server.py
```

The server runs on stdio transport (the MCP standard default).

### Connect with a Python client

```python
import asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

async def main():
    async with Client(PythonStdioTransport("mcp_server/server.py")) as client:
        # List all categories
        result = await client.call_tool("list_categories", {})
        print(result.data)

        # Count rows for REFUND category
        result = await client.call_tool("count_rows", {"category": "REFUND"})
        print(result.data)  # {"count": 2992}

        # Fetch 3 examples with a keyword search
        result = await client.call_tool("get_examples", {
            "n": 3,
            "keyword": "money back"
        })
        for ex in result.data:
            print(ex["intent"], "→", ex["instruction"][:80])

asyncio.run(main())
```

### Available MCP tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `list_categories()` | — | `list[str]` of all 11 categories |
| `list_intents(category)` | `category: str` | `list[str]` of intents in that category |
| `count_rows(category?, intent?)` | optional filters | `{"count": int}` |
| `get_examples(n?, category?, intent?, keyword?, offset?)` | optional filters | `list[dict]` of rows |
| `get_intent_distribution(category)` | `category: str` | `dict[intent → count]` |

---

## Streamlit UI

A browser-based chat interface that wraps the same agent and memory systems as the CLI.

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

**Features:**
- **Session selector** in the sidebar — pick an existing session from the dropdown to resume it, or choose "✚ New session…" to start a fresh one
- **Reasoning steps** — tool calls and observations appear in a collapsible expander inside each assistant message bubble, live-updating as the agent works
- **Profile panel** — for named sessions, the sidebar shows the current user profile as JSON
- **Persistent memory** — same SQLite backend as the CLI; switching to a named session restores that conversation's full history

---

## Architecture

### Graph topology

```
                    START
                      │
               [router_node]          ← cheap model classifies the query
              /        │        \
        "out_of_     "struct"  "unstruct"
         scope"         └────┬────┘
             │               │
       [decline_node]  [agent_node]  ← Hermes-4-70B with tools bound
             │          ↙        ↘
             │    tool calls?    no
             │        │            \
             │  [tool_node]    [profile_update_node]
             │        │                  │
             │        └──→ [agent_node] ─┘  ← ReAct loop repeats until no tool calls
             │
            END                      END
```

- **router_node** — classifies each query as `structured`, `unstructured`, or `out_of_scope` using a small fast model and passes the last 4 messages as context so follow-up queries ("show me 3 more") are never misclassified.
- **agent_node** — the ReAct core. The LLM sees the full message history + system prompt (with user profile injected). It either calls a tool or produces a final answer.
- **tool_node** — LangGraph's built-in `ToolNode`. Executes whichever tool the agent requested and appends the result back to state.
- **profile_update_node** — runs after every agent or decline turn (for named sessions only). Asks the LLM to extract new facts from the conversation and merges them into `profiles/{session_id}.json`.
- **decline_node** — returns a polite refusal without calling the main LLM.

Recursion limit is set once inside `build_graph()` via `.with_config({"recursion_limit": 12})` so it applies to the CLI, LangGraph Studio, and MCP equally.

### Memory — two separate systems

| System | What it stores | Where | Survives restart? |
|--------|---------------|-------|-------------------|
| SqliteSaver (episodic) | Full message history per session | `memory.db` | Yes |
| UserProfileManager (profile) | Distilled facts: name, topics, notes | `profiles/<id>.json` | Yes |

The episodic memory is managed automatically by LangGraph's checkpointer. The user profile is extracted by the LLM after each turn and stored separately, then injected into the system prompt on the next turn.

### Tools

| Tool | Purpose |
|------|---------|
| `get_dataset_schema()` | Returns columns, all categories, all intents, row count |
| `list_categories()` | Returns all 11 category names |
| `list_intents(category)` | Returns intent names within a category |
| `count_rows(category?, intent?)` | Row count with optional filters |
| `get_examples(n, category?, intent?, keyword?, offset?)` | Fetch example rows; `offset` enables pagination |
| `get_intent_distribution(category)` | Intent breakdown as `{intent: count}` dict |
| `summarize_responses(category?, intent?)` | LLM-generated qualitative summary of agent responses |

All tools validate category and intent names against the real schema and return descriptive error dicts on invalid input — the agent is forced to correct itself rather than silently returning empty results.

---

## Model Choice

Two Nebius Token Factory models are used for different roles:

| Role | Model | Cost (in/out per 1M tokens) | Throughput | Rationale |
|------|-------|--------------------------|------------|-----------|
| **Router** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | $0.06 / $0.24 | 60 tok/s | Cheapest text model on Nebius. MoE architecture activates only ~3B parameters per token — fast and cost-effective for a classification task that needs no tool use. |
| **Agent** | `NousResearch/Hermes-4-70B` | $0.13 / $0.40 | 20 tok/s | Fine-tuned specifically on tool-use and function-calling traces (the Hermes series is purpose-built for agentic tasks). Cheapest capable model for reliable multi-step tool calls on Nebius. |

**Why two models?** The router fires on every single message and only needs to output one word (`structured`, `unstructured`, or `out_of_scope`). Using the full 70B agent model for this would be wasteful. The Nemotron router costs ~4× less per token and handles classification reliably, leaving the heavier Hermes model only for turns that require reasoning and tool use.

All models are accessed via [Nebius Token Factory](https://tokenfactory.nebius.com/models) using the OpenAI-compatible API endpoint at `https://api.studio.nebius.ai/v1/`.

---

## LangGraph Studio

To inspect the agent graph visually and send test messages through the Studio UI:

```bash
langgraph dev
```

Then open [smith.langchain.com](https://smith.langchain.com) and navigate to your project. Studio provides its own in-memory persistence — no `memory.db` is written when using Studio.

---

## Project Structure

```
.
├── agent/
│   ├── graph.py       # LangGraph graph assembly
│   ├── llms.py        # Shared LLM instances (Hermes-4-70B + Nemotron-Nano-30B)
│   ├── memory.py      # SqliteSaver setup + UserProfileManager
│   ├── router.py      # Query classification node
│   ├── state.py       # AgentState TypedDict
│   └── tools.py       # All 7 @tool definitions with Pydantic schemas
├── mcp_server/
│   └── server.py      # FastMCP server (5 tools over stdio)
├── data/
│   └── bitext.csv     # Dataset (downloaded via download_data.py)
├── profiles/          # Per-user JSON profiles (git-ignored)
├── main.py            # CLI entry point
├── inspect_memory.py  # SQLite + profile inspection utility
├── download_data.py   # One-time dataset download
├── langgraph.json     # LangGraph Studio config
├── setup.sh           # Mac/Linux setup script
├── setup.ps1          # Windows setup script
├── .env.example       # API key template
└── requirements.txt
```

---

## Inspecting Memory

```bash
# List all sessions in memory.db
python inspect_memory.py --sessions

# Show last 15 messages for a session
python inspect_memory.py --messages alice --n 15

# Show a session's user profile
python inspect_memory.py --profile alice
```
