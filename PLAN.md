# Assignment 3 — Customer Service Data Analyst Agent
## Comprehensive Implementation Plan & Learning Reference

> Due: 29.5.26 | This document is a living reference — update it as we implement.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [LangChain vs LangGraph — The Critical Distinction](#2-langchain-vs-langgraph--the-critical-distinction)
3. [Technology Concepts (The "Why")](#3-technology-concepts-the-why)
   - 3.1 LangGraph & the ReAct Pattern
   - 3.2 The Bitext Dataset
   - 3.3 FastMCP & the MCP Protocol
   - 3.4 LangGraph Memory — Two Separate Systems
4. [Architecture Design](#4-architecture-design)
   - 4.1 Project Structure
   - 4.2 Graph Topology
   - 4.3 Tool Design
   - 4.4 Model Choice (Nebius Token Factory)
5. [Task-by-Task Implementation Plan](#5-task-by-task-implementation-plan)
   - Task 1: Initial Agent (50 pts)
   - Task 2a: Conversation Memory (20 pts)
   - Task 2b: User Profile (10 pts)
   - Task 3: MCP Server (20 pts)
   - Bonus A: Streamlit UI (+10 pts)
   - Bonus B: Query Recommender (+10 pts)
6. [Key Learning Concepts](#6-key-learning-concepts-transferable-to-any-project)

---

## 1. The Big Picture

```
User question (CLI or Streamlit)
        ↓
  [LangGraph ReAct Agent]
        ↓
  router_node → out-of-scope? → polite decline
        ↓
  agent_node (LLM thinks: "which tool do I need?")
        ↓
  tool_node (executes pandas/summarization tools)
        ↓ ← ← ← loop back ← ← ←
  agent_node (LLM sees result, decides: done or call another tool?)
        ↓
  Final answer (printed with full reasoning trace)
        ↓
  [SqliteSaver persists conversation state to disk]
  [User profile JSON updated if new facts emerged]
        ↓
  [FastMCP server exposes 3+ tools over MCP protocol]
```

---

## 2. LangChain vs LangGraph — The Critical Distinction

This is probably the most important thing to understand before writing a single line of code.

### LangChain — Chains, not Graphs

LangChain is a library of building blocks for LLM applications:
- Prompt templates, output parsers, document loaders, retrievers
- "Chains" — linear sequences: input → LLM → parse → output
- "Agents" (the old LCEL style) — a while loop that calls an LLM, checks if it wants to use a tool, calls the tool, feeds the result back

The old LangChain agent was essentially:

```python
# Old LangChain agent — implicit loop, hard to control
agent = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE)
result = agent.run("How many refund requests?")
# You can't easily intercept mid-loop, add custom routing, or persist state
```

**LangChain's limitation for agents:** The loop is hidden inside the library. You can't add a custom router node, easily inspect intermediate states, inject memory between steps, or define fallback paths.

### LangGraph — Explicit State Machines

LangGraph was built specifically to solve this. It models agent execution as a **directed graph**:
- **Nodes** = Python functions (each does one job)
- **Edges** = control flow between nodes (can be conditional)
- **State** = a TypedDict that flows through every node and is mutated along the way
- **Checkpointer** = saves the State after every node (enables persistence, time-travel, resumption)

```python
# LangGraph — explicit, inspectable, controllable
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("router", route_query, {
    "structured": "agent",
    "out_of_scope": "decline",
})
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
```

### The Relationship Between the Two

LangGraph is a separate package (`langgraph`) that **uses** LangChain's LLM abstractions internally:
- You still use `langchain_openai.ChatOpenAI` to instantiate the LLM
- You still use `@tool` from `langchain_core.tools` to define tools
- LangGraph adds the graph runtime on top

**Mental model:**
```
LangChain = LEGO bricks (LLMs, prompts, tools, retrievers)
LangGraph  = the instruction manual + the baseplate that holds it all together as a stateful machine
```

### Why LangGraph for this assignment?

| Need | LangChain alone | LangGraph |
|---|---|---|
| Custom router before agent | Hard — you'd have to subclass | Trivial — just add a node |
| Persistent memory across restarts | Manual, fragile | Built-in with checkpointers |
| Max iterations with graceful fallback | You'd hand-write the loop | `recursion_limit` config |
| Inspect every step | Not easy | Every node's I/O is logged |
| Follow-up queries ("show 3 more") | You'd manage history manually | State carries `messages` automatically |

### When to use LangChain without LangGraph

For simple, linear pipelines (summarize this doc, extract these fields, classify this text) — plain LangChain LCEL is simpler and has less overhead. Use LangGraph when you need:
- Multi-step loops that can vary per query
- Persistent state
- Custom routing/branching logic
- Multiple agents talking to each other

---

## 3. Technology Concepts (The "Why")

### 3.1 LangGraph & the ReAct Pattern

**ReAct** (Reason + Act) is a prompting pattern where an LLM alternates between:
1. **Thinking** — "I need to know how many rows match 'refund'"
2. **Acting** — calls `count_rows(intent="track_refund")`
3. **Observing** — sees `1012`, then thinks again: "That's my answer"

In LangGraph, this is implemented as:
- `agent_node`: binds tools to the LLM, invokes it, gets a response that may include tool_call requests
- `tool_node`: if the LLM response has tool calls, executes them and appends the results to `messages`
- Conditional edge: "does the latest message have tool calls? Yes → go to tool_node. No → END"

**The loop:**
```
agent_node → (tool calls present?) → tool_node → agent_node → ... → agent_node → END
```

**Max iterations safety valve:**
```python
graph = workflow.compile(checkpointer=checkpointer)
# At invocation time:
result = graph.invoke(input, config={
    "configurable": {"thread_id": session_id},
    "recursion_limit": 12   # After 12 node visits total, raise an error we catch
})
```

We wrap the invoke in a try/except for `GraphRecursionError` and return a graceful fallback message.

**Docs:**
- https://langchain-ai.github.io/langgraph/concepts/
- https://dev.to/agentsindex/langgraph-tutorial-build-a-working-react-agent-with-the-v10-api-3bc1

---

### 3.2 The Bitext Dataset

**Source:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

**Size:** 26,872 rows, 19.2 MB, ~1,000 rows per intent

**Columns:**

| Column | Type | Description |
|---|---|---|
| `flags` | string | Linguistic variation tags (B=basic, I=interrogative, P=polite, Q=colloquial, etc.) |
| `instruction` | string | Customer's message (e.g. "I need to cancel my order #1234") |
| `category` | string | High-level group — 10 categories |
| `intent` | string | Specific action — 27 intents |
| `response` | string | The ideal agent response |

**10 categories and their intents:**
```
ACCOUNT:          create_account, delete_account, edit_account, switch_account
CANCELLATION_FEE: check_cancellation_fee
DELIVERY:         delivery_options
FEEDBACK:         complaint, review
INVOICE:          check_invoice, get_invoice
NEWSLETTER:       newsletter_subscription
ORDER:            cancel_order, change_order, place_order
PAYMENT:          check_payment_methods, payment_issue
REFUND:           check_refund_policy, track_refund
SHIPPING_ADDRESS: change_shipping_address, set_up_shipping_address
```

**Important for tool design:** Note that "refund requests" in natural language maps to the `track_refund` intent (not `check_refund_policy`). The agent needs to reason about this mapping. The test query "Show me examples of people wanting their money back" has no explicit `REFUND` keyword — the agent must use keyword search or semantic reasoning to find the right intent.

---

### 3.3 FastMCP & the MCP Protocol

**MCP (Model Context Protocol)** is a standard protocol (like REST, but purpose-built for AI tools) that lets any MCP-compatible client (Claude Desktop, Cursor, your agent) call tools exposed by any MCP server — without the server knowing anything about the client.

**FastMCP** is a Python library that makes building MCP servers as simple as decorating functions:

```python
from fastmcp import FastMCP

mcp = FastMCP("Bitext Analyst")

@mcp.tool
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count rows in the Bitext dataset, optionally filtered by category or intent."""
    ...

if __name__ == "__main__":
    mcp.run()  # starts the server (stdio transport by default)
```

FastMCP automatically:
- Generates the JSON schema from your type hints (so MCP clients know how to call it)
- Validates incoming parameters with Pydantic
- Handles the MCP handshake and lifecycle

**Architecture note:** In this project, you're doing something architecturally elegant — your LangGraph agent already has tools defined. The MCP server re-exposes those same tool functions through the MCP protocol, so any external client can call them independently of the agent.

**FastMCP docs:** https://gofastmcp.com/getting-started/welcome | https://gofastmcp.com/servers/tools

---

### 3.4 LangGraph Memory — Two Separate Systems

The assignment requires two distinct memory systems. Understanding the difference is critical.

#### Episodic Memory (Task 2a) — SqliteSaver

LangGraph's checkpointer saves the **entire AgentState** (including all messages) to SQLite after every node execution. The key is the `thread_id`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("./memory.db")
graph = workflow.compile(checkpointer=checkpointer)

# Each invocation uses a thread_id — same ID = same conversation restored
config = {"configurable": {"thread_id": "alice_session"}}
result = graph.invoke({"messages": [HumanMessage("Show me 3 refund examples")]}, config)
# Later (even after restart):
result = graph.invoke({"messages": [HumanMessage("Show me 3 more")]}, config)
# The agent sees the previous 3 examples in its message history and correctly offsets
```

**Important:** `MemorySaver` (in-memory) is for development only — it loses everything on restart. Use `SqliteSaver` for persistence.

**Install:** `pip install langgraph-checkpoint-sqlite`

#### User Profile (Task 2b) — Separate JSON Files

This is NOT stored in the LangGraph checkpointer. It's a separate file per user:

```
profiles/
  alice.json    → {"name": "Alice", "frequent_topics": ["REFUND", "ACCOUNT"], ...}
  bob.json      → {"name": null, "frequent_topics": ["ORDER"], ...}
```

**How it works:**
1. After the agent produces a final answer, a `profile_update_node` runs
2. It calls the LLM with: "Given this conversation, what new facts did we learn about the user? Return JSON."
3. The JSON is merged into the user's profile file
4. On the next turn, the profile is loaded and injected into the system prompt

**The distinction in one sentence:**
- Episodic memory = the full transcript, managed by LangGraph automatically
- User profile = distilled facts about the person, managed by your code explicitly

#### Memory Technique Taxonomy (from the reference repo)

The assignment points to https://github.com/NirDiamant/Agent_Memory_Techniques which catalogs 30 memory approaches. For this project, we're combining three of them:

| Technique | What it is | How we use it |
|---|---|---|
| **Summary Memory** | Instead of storing every message, periodically compress the conversation into key facts | The profile update node does this — it reads recent messages and distills them into a JSON profile |
| **Entity Memory** | Track named entities and facts about them ("user cares about REFUND data") | The profile's `frequent_topics` and `preferences` fields |
| **Cross-Session Memory** | Persist facts about the user across multiple sessions/restarts | The per-user JSON file and SqliteSaver both survive restarts |

The other 27 techniques (vector store memory, knowledge graphs, memory decay, etc.) are more relevant for long-running production systems. For this assignment, the three above are exactly what's needed.

**Docs:**
- https://github.com/NirDiamant/Agent_Memory_Techniques/tree/main/all_techniques
- https://langchain-tutorials.github.io/langgraph-memory-implementation-complete-guide/
- https://medium.com/@princekrampah/external-persistent-memory-for-agents-building-robust-applications-with-langgraph-8415b170beef

---

## 4. Architecture Design

### 4.1 Project Structure

```
customer-service-agent/
├── main.py                  # CLI entry: python main.py --session alice
├── streamlit_app.py         # Bonus A: Streamlit chat UI
├── download_data.py         # One-time dataset download from HuggingFace
├── langgraph.json           # LangGraph Studio config (points to agent/graph.py:graph)
├── setup.sh                 # Mac/Linux one-command setup
├── setup.ps1                # Windows one-command setup
├── .env.example             # Template for NEBIUS_API_KEY + LANGSMITH_API_KEY
│
├── agent/
│   ├── __init__.py
│   ├── state.py             # AgentState TypedDict definition
│   ├── graph.py             # LangGraph graph assembly and compilation
│   ├── router.py            # Router node (classifies query type, with conversation context)
│   ├── tools.py             # All @tool definitions with Pydantic schemas
│   ├── llms.py              # Shared LLM instances (agent_llm + router_llm)
│   └── memory.py            # SqliteSaver setup + UserProfileManager  [Task 2]
│
├── mcp_server/
│   ├── __init__.py
│   └── server.py            # FastMCP server exposing 3+ tools
│
├── data/
│   └── bitext.csv           # Dataset (downloaded at setup)
│
├── profiles/                # Per-user profile JSON files (task 2b)
├── memory.db                # SqliteSaver SQLite file (task 2a)
│
├── requirements.txt
└── README.md
```

---

### 4.2 Graph Topology

```
            START
              ↓
        [router_node]         ← classifies: structured / unstructured / out_of_scope
         /    |    \
        /     |     \
   "out"   "struct" "unstruct"
     ↓        ↓         ↓
[decline]  [agent_node] ←──────────────────┐
              ↓                            │
     has tool calls?                       │
       /           \                       │
    yes             no                     │
     ↓               ↓                    │
[tool_node]    [profile_update_node]       │
     └──────────────────────────────────→─┘
              ↓ (when LLM produces final answer, or max iterations hit)
             END
```

**Node responsibilities:**
- `router_node`: Sends query to a small/fast LLM with a strict classification prompt. Returns "structured", "unstructured", or "out_of_scope". Sets `state["query_type"]`.
- `agent_node`: The ReAct core. LLM has all tools bound. Sees the full message history. Produces either a tool call or a final response.
- `tool_node`: Built-in `ToolNode(tools)` from LangGraph. Executes whichever tool the agent requested and appends a `ToolMessage` to the state.
- `profile_update_node`: Reads the completed conversation, extracts new user facts, writes to `profiles/{session_id}.json`.
- `decline_node`: Returns a polite refusal. Does NOT call the LLM with general knowledge.

---

### 4.3 Tool Design

> "A few well-designed tools beat many poorly described ones." — T. Braude

**Rule:** The tool's description IS its interface to the LLM. If the description doesn't tell the LLM exactly when to use it and what to pass, the agent will guess wrong.

| Tool | Input Schema | Returns | Query type |
|---|---|---|---|
| `get_dataset_schema()` | none | Column names + category/intent list | Meta |
| `list_categories()` | none | `list[str]` of all categories | Structured |
| `list_intents(category)` | `category: str` | `list[str]` of intents in that category | Structured |
| `count_rows(category?, intent?)` | optional filters | `int` count | Structured |
| `get_examples(n, category?, intent?, keyword?)` | n + optional filters | `list[dict]` of rows | Structured |
| `get_intent_distribution(category)` | `category: str` | `dict[intent -> count]` | Structured |
| `summarize_responses(category?, intent?)` | optional filters | `str` LLM-generated summary | Unstructured |

**Multi-step reasoning examples:**

```
Query: "How many refund requests?"
  → agent calls list_intents("REFUND") to confirm intent names
  → agent calls count_rows(intent="track_refund")
  → agent: "There are 1,012 refund requests"

Query: "Show me examples of people wanting their money back"
  → agent calls get_examples(n=5, keyword="refund")  OR
  → agent calls list_intents("REFUND"), then get_examples(n=5, intent="track_refund")

Query: "What is the distribution of intents in ACCOUNT?"
  → agent calls get_intent_distribution("ACCOUNT")
  → agent formats the dict into a readable table

Query: "Summarize how agents respond to cancellation requests"
  → router classifies as "unstructured"
  → agent calls summarize_responses(intent="cancel_order")
  → LLM summarizes the response column text
```

**Pydantic input schema example:**

```python
from pydantic import BaseModel, Field

class GetExamplesInput(BaseModel):
    n: int = Field(default=5, ge=1, le=50, description="Number of examples to return (1-50)")
    category: str | None = Field(
        default=None,
        description="Filter by category name (e.g. 'REFUND', 'ACCOUNT'). Case-insensitive."
    )
    intent: str | None = Field(
        default=None,
        description="Filter by intent name (e.g. 'track_refund'). Case-insensitive."
    )
    keyword: str | None = Field(
        default=None,
        description="Keyword to search within customer instructions. Use when the user describes "
                    "a topic in plain language rather than naming a specific category or intent."
    )
```

---

### 4.4 Model Choice (Nebius Token Factory)

Nebius Token Factory provides 60+ open-source models via an OpenAI-compatible API.

**Actual setup (verified against live Nebius pricing page):**

| Role | Model | Price In | Price Out | Throughput | Reason |
|---|---|---|---|---|---|
| **Router** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | $0.06/1M | $0.24/1M | 60 Tok/s | Cheapest text model; MoE with ~3B active params; pure text so no wasted multimodal capacity |
| **Agent** | `NousResearch/Hermes-4-70B` | $0.13/1M | $0.40/1M | 20 Tok/s | Specifically fine-tuned on verified tool-use / function-calling traces; cheapest capable model for agentic tasks |

**Note on models that don't exist on Nebius:** `meta-llama/Llama-3.1-8B-Instruct` returns 404 — Nebius doesn't host this variant.

**Note on throughput:** Hermes-4-70B at 20 Tok/s is noticeably slower than alternatives (DeepSeek-V3.2 at 71 Tok/s). Accepted trade-off for Task 1 to prioritise tool-call correctness. Can swap to DeepSeek-V3.2 if speed becomes an issue during development.

**How to instantiate (via LangChain's OpenAI wrapper — see `agent/llms.py`):**
```python
from langchain_openai import ChatOpenAI

agent_llm = ChatOpenAI(
    model="NousResearch/Hermes-4-70B",
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
    temperature=0,
)

router_llm = ChatOpenAI(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
    temperature=0,
)
```

**Models reference:** https://tokenfactory.nebius.com/models

---

## 5. Task-by-Task Implementation Plan

### Task 1 — Initial Agent (50 pts)

**Grading:** query router (15) + tools with Pydantic schemas (15) + multi-step reasoning (10) + CLI with reasoning output (5) + max iterations fallback (5)

#### Step 1.1 — Download & load the dataset

```python
# agent/tools.py (top of file, module-level)
import pandas as pd

df = pd.read_csv("data/bitext.csv")
CATEGORIES = df["category"].str.upper().unique().tolist()
INTENTS = df["intent"].str.lower().unique().tolist()
```

Load once at import time. The 19MB dataset fits easily in memory.

#### Step 1.2 — Define AgentState

```python
# agent/state.py
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str        # "structured" | "unstructured" | "out_of_scope"
    iterations: int        # Manual iteration counter (backup to recursion_limit)
    user_profile: dict     # Loaded at start, updated at end
```

**Why `add_messages` annotation?** This tells LangGraph to *append* new messages to the list rather than *replace* it. Without this, every node that writes to `messages` would overwrite the history.

#### Step 1.3 — Router node

```python
# agent/router.py
ROUTER_PROMPT = """You are a query classifier for a customer service data analyst agent.
The agent has access to the Bitext Customer Service dataset which contains:
- Customer support queries and responses
- Categories: {categories}
- Intents: {intents}

Classify the user's question as exactly one of:
- "structured": Has a concrete, data-driven answer (counts, lists, examples, distributions)
- "unstructured": Requires summarization or qualitative analysis of the data
- "out_of_scope": Unrelated to the Bitext customer service dataset

Return ONLY the classification word, nothing else.

User question: {question}"""

def router_node(state: AgentState) -> AgentState:
    """Classifies the incoming query before agent tool selection."""
    question = state["messages"][-1].content
    prompt = ROUTER_PROMPT.format(
        categories=CATEGORIES, intents=INTENTS, question=question
    )
    response = router_llm.invoke(prompt)
    classification = response.content.strip().lower()
    if classification not in ("structured", "unstructured", "out_of_scope"):
        classification = "structured"  # safe fallback
    return {"query_type": classification}
```

#### Step 1.4 — Tools with Pydantic schemas

See Section 4.3 above for the full tool list. Key implementation notes:
- Use `@tool` decorator with explicit `args_schema` pointing to a Pydantic `BaseModel`
- Filter with `df[df["category"].str.upper() == category.upper()]` for case-insensitive matching
- For keyword search: `df[df["instruction"].str.contains(keyword, case=False, na=False)]`
- For summarization tools: call the LLM directly inside the tool function (not the agent)

#### Step 1.5 — Assemble the graph

```python
# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("decline", decline_node)
    workflow.add_node("profile_update", profile_update_node)

    # Entry
    workflow.set_entry_point("router")

    # Router → branch
    workflow.add_conditional_edges("router", lambda s: s["query_type"], {
        "structured":   "agent",
        "unstructured": "agent",
        "out_of_scope": "decline",
    })

    # Agent → branch (has tool calls or done?)
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end":   "profile_update",
    })

    # Tools loop back to agent
    workflow.add_edge("tools", "agent")

    # Terminal nodes
    workflow.add_edge("decline", END)
    workflow.add_edge("profile_update", END)

    return workflow.compile(checkpointer=checkpointer)
```

#### Step 1.6 — CLI interface

```python
# main.py
import argparse
from langgraph.errors import GraphRecursionError

parser = argparse.ArgumentParser()
parser.add_argument("--session", default="default")
args = parser.parse_args()

graph = build_graph(checkpointer=checkpointer)
config = {"configurable": {"thread_id": args.session}, "recursion_limit": 12}

print(f"Session: {args.session}. Type 'quit' to exit.\n")
while True:
    query = input("You: ").strip()
    if query.lower() in ("quit", "exit"):
        break
    try:
        for event in graph.stream({"messages": [("human", query)]}, config, stream_mode="values"):
            # Print tool calls and observations as they happen
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"  [TOOL CALL] {tc['name']}({tc['args']})")
            elif last_msg.type == "tool":
                print(f"  [OBSERVATION] {last_msg.content[:200]}")
        print(f"\nAgent: {event['messages'][-1].content}\n")
    except GraphRecursionError:
        print("Agent: I've reached my reasoning limit on this question. Please try rephrasing.\n")
```

---

### Task 2a — Conversation Memory (20 pts)

#### How SqliteSaver works

```python
# agent/memory.py
from langgraph.checkpoint.sqlite import SqliteSaver

def get_checkpointer() -> SqliteSaver:
    """Returns a SqliteSaver connected to the local memory.db file."""
    return SqliteSaver.from_conn_string("./memory.db")
```

The `thread_id` in the config is the session identifier. Every time you call `graph.invoke()` or `graph.stream()` with the same `thread_id`, LangGraph:
1. Loads the latest checkpoint for that thread from SQLite
2. Appends your new message to the existing history
3. Runs the graph
4. Saves the new state back to SQLite

This is why follow-up queries work automatically — the agent always sees the full conversation history.

#### Follow-up query example trace:

```
Turn 1: "Show me 3 examples from REFUND"
  → agent calls get_examples(n=3, category="REFUND")
  → returns 3 rows
  → state saved to memory.db with thread_id="alice"

Turn 2: "Show me 3 more"
  → LangGraph loads thread "alice" — messages include Turn 1's results
  → agent reads history, understands "3 more" = examples 4-6 of REFUND
  → agent calls get_examples(n=3, category="REFUND") and the LLM knows to offset
  (or: the tool could support an `offset` parameter for this exact case)
```

---

### Task 2b — User Profile (10 pts)

#### UserProfileManager

```python
# agent/memory.py (continued)
import json
from pathlib import Path

class UserProfileManager:
    """Manages per-user persistent profiles stored as JSON files."""

    PROFILES_DIR = Path("./profiles")

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.PROFILES_DIR.mkdir(exist_ok=True)
        self.path = self.PROFILES_DIR / f"{session_id}.json"

    def load(self) -> dict:
        """Load profile, returning empty profile if not found."""
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"name": None, "frequent_topics": [], "preferences": {}, "notes": []}

    def save(self, profile: dict) -> None:
        """Persist the profile to disk."""
        self.path.write_text(json.dumps(profile, indent=2))

    def update_from_conversation(self, messages: list, llm) -> dict:
        """Use the LLM to extract new facts and merge into the profile."""
        current = self.load()
        conversation_text = "\n".join(
            f"{m.type}: {m.content[:300]}" for m in messages[-10:]
        )
        prompt = f"""Given this conversation and the current user profile, 
extract any new facts about the user (name, interests, frequently-asked topics, preferences).
Return a JSON object with only the fields that changed or were added.

Current profile: {json.dumps(current)}

Recent conversation:
{conversation_text}

Return ONLY valid JSON, no explanation."""
        response = llm.invoke(prompt)
        try:
            updates = json.loads(response.content)
            current.update(updates)
            self.save(current)
        except json.JSONDecodeError:
            pass  # If LLM returned malformed JSON, skip update silently
        return current
```

#### Profile injection into the system prompt

```python
def get_system_prompt(user_profile: dict) -> str:
    profile_context = ""
    if user_profile.get("name"):
        profile_context += f"The user's name is {user_profile['name']}. "
    if user_profile.get("frequent_topics"):
        profile_context += f"They frequently ask about: {', '.join(user_profile['frequent_topics'])}. "
    return f"""You are a data analyst agent for the Bitext Customer Service dataset.
{profile_context}
Answer questions about the dataset using your tools. Be concise and factual.
For out-of-scope questions, politely decline — do not answer from general knowledge."""
```

---

### Task 3 — FastMCP Server (20 pts)

```python
# mcp_server/server.py
from fastmcp import FastMCP
from agent.tools import list_categories, count_rows, get_examples, get_intent_distribution

mcp = FastMCP(
    name="Bitext Customer Service Analyst",
    instructions="Tools for querying the Bitext customer support dataset."
)

@mcp.tool
def mcp_list_categories() -> list[str]:
    """List all available categories in the Bitext customer service dataset."""
    return list_categories.invoke({})

@mcp.tool
def mcp_count_rows(
    category: str | None = None,
    intent: str | None = None
) -> int:
    """Count rows in the dataset. Optionally filter by category (e.g. 'REFUND') or intent (e.g. 'track_refund')."""
    return count_rows.invoke({"category": category, "intent": intent})

@mcp.tool
def mcp_get_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None
) -> list[dict]:
    """Get example rows from the dataset. Filter by category, intent, or keyword search."""
    return get_examples.invoke({"n": n, "category": category, "intent": intent, "keyword": keyword})

@mcp.tool
def mcp_get_intent_distribution(category: str) -> dict:
    """Get the count of each intent within a given category."""
    return get_intent_distribution.invoke({"category": category})

if __name__ == "__main__":
    mcp.run()
```

**Starting the server:**
```bash
python mcp_server/server.py
```

**Connecting a client (README example):**
```python
from fastmcp import Client

async def main():
    async with Client("python mcp_server/server.py") as client:
        result = await client.call_tool("mcp_list_categories", {})
        print(result)
```

---

### Bonus A — Streamlit UI (+10 pts)

```python
# streamlit_app.py
import streamlit as st
from agent.graph import build_graph
from agent.memory import get_checkpointer

st.title("Customer Service Data Analyst")

# Sidebar: session management
with st.sidebar:
    session_id = st.text_input("Session ID", value="default")
    st.caption("Same session ID = same conversation restored")

# Initialize graph (cached so it's not rebuilt on every interaction)
@st.cache_resource
def get_graph():
    return build_graph(checkpointer=get_checkpointer())

graph = get_graph()

# Chat interface
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the customer service data..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 12}

    with st.chat_message("assistant"):
        reasoning_container = st.expander("Reasoning steps", expanded=False)
        final_answer = ""

        for event in graph.stream({"messages": [("human", prompt)]}, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    reasoning_container.write(f"**Tool call:** `{tc['name']}({tc['args']})`")
            elif last_msg.type == "tool":
                reasoning_container.write(f"**Observation:** {last_msg.content[:300]}")
            else:
                final_answer = last_msg.content

        st.write(final_answer)
        st.session_state.display_messages.append({"role": "assistant", "content": final_answer})
```

---

### Bonus B — Query Recommender (+10 pts)

**How it works:**
1. Router detects a "recommendation request" (e.g. "what should I query next?")
2. A `recommend_node` reads the user profile + recent conversation
3. LLM generates a suggested query — stored in state as `pending_suggestion`, NOT executed
4. Agent presents the suggestion to the user
5. Next user turn: if they confirm → execute; if they refine → loop

**State additions:**
```python
class AgentState(TypedDict):
    # ... existing fields ...
    pending_suggestion: str | None   # Suggested query awaiting user confirmation
    awaiting_confirmation: bool      # True when agent is waiting for yes/no
```

**Example flow:**
```
User:  "What should I query next?"
Router: classifies as "recommendation"
Recommend node: reads profile (frequent_topics=["REFUND"])
LLM suggests: "Show me the distribution of intents in the REFUND category"
Agent: "Based on your interest in REFUND data, I suggest: 
        'Show the distribution of intents in REFUND'. Should I go ahead?"
State: pending_suggestion = "distribution of REFUND", awaiting_confirmation = True

User:  "I'd rather see examples"
Agent: "Understood. I suggest: 'Show 5 examples from REFUND'. Should I go ahead?"
State: pending_suggestion = "5 examples from REFUND", awaiting_confirmation = True

User:  "Yes"
Agent: [executes get_examples(n=5, category="REFUND")]
State: pending_suggestion = None, awaiting_confirmation = False
```

---

## 6. Key Learning Concepts (Transferable to Any Project)

| Concept | What You'll Understand | Apply It To |
|---|---|---|
| **LangGraph State** | Shared typed dict flowing through every node | Any multi-step agent |
| **add_messages reducer** | How to append vs replace list fields in state | Custom state fields |
| **ReAct loop** | Think → Act → Observe → repeat | All tool-using LLM agents |
| **Router node** | Classifying input before expensive processing | Any pipeline with branches |
| **Pydantic tool schemas** | The schema IS the LLM's interface | Every LLM agent framework |
| **SqliteSaver + thread_id** | Persistent sessions without a server | Chat apps, tutors, assistants |
| **add_messages vs user profile** | Full history vs distilled facts | Any personalized AI product |
| **MCP protocol** | Exposing tools to any AI client | Building composable AI tooling |
| **Dual-model strategy** | Cheap model for routing, powerful for reasoning | Cost optimization in prod |
| **GraphRecursionError handling** | Graceful fallback when agents loop | Production safety |

---

## Implementation Checklist

### Task 1 ✅ COMPLETE
- [x] Download dataset to `data/bitext.csv` (via `download_data.py`)
- [x] `agent/state.py` — AgentState TypedDict with `add_messages`, `query_type`, `user_profile`
- [x] `agent/llms.py` — shared LLM config: Hermes-4-70B (agent) + Nemotron-3-Nano-30B-A3B (router)
- [x] `agent/tools.py` — all 7 tools with Pydantic schemas (get_dataset_schema, list_categories, list_intents, count_rows, get_examples with offset, get_intent_distribution, summarize_responses)
- [x] `agent/router.py` — router node with classification prompt + conversation context (fixes follow-up queries)
- [x] `agent/graph.py` — full graph: router → agent ↔ tools → END, decline path, no baked-in checkpointer (Studio-compatible)
- [x] `main.py` — CLI loop with [TOOL]/[OBS] streaming output, MemorySaver for within-session memory, max iterations handling
- [x] `langgraph.json` — LangGraph Studio config, verified working with `langgraph dev`
- [x] `.gitignore`, `setup.sh`, `setup.ps1`, `.env.example` — cross-platform project scaffolding
- [x] LangSmith tracing verified working

**Known limitation (to fix in Task 2a):** messages list grows unboundedly — add `trim_messages` when adding SqliteSaver.

### Task 2 ✅ COMPLETE
- [x] `agent/memory.py` — SqliteSaver setup (`get_checkpointer()`) + `UserProfileManager` class
- [x] Replace `MemorySaver` in `main.py` with `SqliteSaver` via `get_checkpointer()` — memory now survives restarts
- [x] Added `_MAX_MESSAGES = 20` trim in `agent_node` to cap context window growth
- [x] `profile_update_node` in `graph.py` — LLM extracts user facts after each answer, writes `profiles/{session_id}.json`
- [x] `user_profile` loaded from JSON and injected into system prompt in `agent_node`
- [x] Graph topology updated: agent → profile_update → END; decline → profile_update → END (profile runs on all paths)
- [x] `profile_enabled` flag — profile only tracked for named sessions (`--session alice`); anonymous sessions skip entirely
- [x] Tested: quit and restart CLI with same `--session` — prior conversation restored from memory.db
- [x] Tested: `profiles/{session_id}.json` written and updated; `--show-profile` and `--list-sessions` CLI commands verified

**Post-Task-2 refinements:**
- [x] `count_rows` validates category/intent against schema — returns error dict instead of silent 0
- [x] `get_examples` same validation added — returns error dict instead of empty list
- [x] System prompt Rule 4: sum counts from conversation history, never invent numbers
- [x] System prompt Rule 5 (vary intent on "different") removed — offset inherently gives different rows; was incorrect guidance
- [x] Router defensive guards: empty message → out_of_scope; unknown classification → structured (not crash)

### Task 3 ✅ COMPLETE
- [x] `mcp_server/server.py` — FastMCP server with 5 tools (list_categories, list_intents, count_rows, get_examples, get_intent_distribution)
- [x] All tools share the in-memory DataFrame from `agent/tools.py` — no double load
- [x] Input validation consistent with agent tools (error dicts on bad category/intent)
- [x] README section: starting the server + Python client connection example

### Bonus
- [ ] `streamlit_app.py` — Streamlit chat UI
- [ ] Query recommender: `pending_suggestion` state + recommend node

### Deliverables
- [x] `requirements.txt` — all dependencies covered (validated against actual imports)
- [x] `README.md` — setup (≤5 min), CLI usage, MCP client example, architecture overview, model choice justification
- [ ] GitHub repo (solo submission — repo name: `eden-lalum-customer-service-agent` or similar)
