from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llms import agent_llm
from agent.memory import UserProfileManager
from agent.router import router_node
from agent.state import AgentState
from agent.tools import TOOLS

_RECURSION_LIMIT = 12
_MAX_MESSAGES = 20


def _get_system_prompt(user_profile: dict) -> str:
    base = """You are a data analyst assistant with tools to query a structured tabular dataset.

Rules you must follow:
1. If you are unsure of the dataset's structure, call get_dataset_schema() first — it returns columns, row count, and available filter values.
2. NEVER assume what categories or intents exist. Verify a category exists before filtering by it. To count ALL rows in a category call count_rows(category=...) directly — do NOT call list_intents() first. Only call list_intents() when you need to filter by a specific intent.
3. Always use your tools for data questions — never guess at counts, names, or distributions.
4. When asked to total or sum counts from earlier in this conversation, add the exact numbers that tools already returned in this session. Never invent, guess, or round. Cross-session aggregation is not supported.
5. If the user states a preference or expresses a like/dislike ("I like X", "I prefer Y", "I enjoy Z"), acknowledge it conversationally. Do NOT call any tools for preference statements.
6. If the user asks what you know or remember about them, summarize the User context section of this prompt — that is what you know. Do not claim you have no information.
7. When the user asks what to query next (e.g. "what should I query?", "suggest something", "what else can I ask?", "what would be interesting?"):
   a. Look at the User context (profile) — pick the most-mentioned topic from frequent_topics and factor in their stated preferences (e.g. if style=examples, suggest examples not counts).
   b. ALWAYS name the specific profile signal driving your suggestion. Say which topic or preference you are drawing from.
   c. Describe the suggestion in fluent plain English. NEVER write code, function names, or parameter syntax. Good: "see 5 examples from the REFUND category". Bad: "get_examples(category='REFUND', n=5)".
   d. Use this exact pattern — two sentences: first explain the profile signal, then state the suggestion and ask:
      "Based on your interest in [topic from profile], you might want to [plain English description]. Should I go ahead?"
   e. If the user confirms ("yes", "sure", "go ahead", "do it") → execute with tools.
   f. If the user refines the topic or scope ("I'd rather see examples", "make it about shipping") → acknowledge, re-suggest with the refinement in plain English, ask again.
      NOTE: if the user makes a preference statement ("I like short answers", "I prefer concise") treat it as Rule 5 — acknowledge the preference and then re-ask if they want the suggestion run.
   g. If the user asks a completely unrelated data question → treat as a new question, drop the suggestion.
8. Be concise and factual.
9. If a question is clearly unrelated to the dataset, politely decline."""

    extras = []
    if user_profile.get("name"):
        extras.append(f"The user's name is {user_profile['name']}.")
    # Only surface topics mentioned 2+ times — single mentions are not yet "frequent"
    # Guard against stale list format from old SQLite checkpoints
    raw_topics = user_profile.get("frequent_topics", {})
    if isinstance(raw_topics, dict):
        frequent = [t for t, count in raw_topics.items() if count >= 2]
    else:
        frequent = list(raw_topics)  # old list format — show as-is during migration
    if frequent:
        extras.append(f"They frequently ask about: {', '.join(frequent)}.")
    if user_profile.get("preferences"):
        prefs = "; ".join(f"{k}: {v}" for k, v in list(user_profile["preferences"].items())[:3])
        extras.append(f"Preferences: {prefs}.")

    if extras:
        return base + "\n\nUser context: " + " ".join(extras)
    return base


_agent_with_tools = agent_llm.bind_tools(TOOLS)


def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    session_id = config.get("configurable", {}).get("thread_id", "default")
    profile = state.get("user_profile") or UserProfileManager(session_id).load()
    messages = state["messages"][-_MAX_MESSAGES:]
    response = _agent_with_tools.invoke(
        [SystemMessage(content=_get_system_prompt(profile))] + messages
    )
    return {"messages": [response]}


def decline_node(state: AgentState) -> AgentState:
    return {
        "messages": [
            AIMessage(
                content=(
                    "I can only answer questions about the loaded dataset. "
                    "Try asking about its structure, categories, examples, or patterns."
                )
            )
        ]
    }


def profile_update_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Updates the user profile for named sessions.
    Skips anonymous sessions (no --session flag) entirely."""
    messages = state["messages"]

    configurable = config.get("configurable", {})
    if not configurable.get("profile_enabled", False):
        return {}  # Anonymous/unnamed session — no profile tracking.

    session_id = configurable.get("thread_id", "default")
    manager = UserProfileManager(session_id)
    updated = manager.update_from_conversation(messages, agent_llm)
    return {"user_profile": updated}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("decline", decline_node)
    workflow.add_node("profile_update", profile_update_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s["query_type"],
        {"structured": "agent", "unstructured": "agent", "out_of_scope": "decline"},
    )
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "profile_update"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("decline", END)
    workflow.add_edge("profile_update", END)

    compiled = workflow.compile(checkpointer=checkpointer)
    # Set recursion limit once here — applies to CLI, Studio, MCP, everything.
    return compiled.with_config({"recursion_limit": _RECURSION_LIMIT})


# Module-level graph for langgraph dev / Studio (no checkpointer — Studio provides its own).
graph = build_graph()
