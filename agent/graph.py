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
2. NEVER assume what categories, intents, or field values exist. Before filtering by any specific value, verify it exists first by calling list_categories() or list_intents(). Only use values those tools actually return.
3. Always use your tools for data questions — never guess at counts, names, or distributions.
4. When asked to total or sum counts from earlier in this conversation, add the exact numbers that tools already returned in this session. Never invent, guess, or round. Cross-session aggregation is not supported.
5. Be concise and factual.
6. If a question is clearly unrelated to the dataset, politely decline."""

    extras = []
    if user_profile.get("name"):
        extras.append(f"The user's name is {user_profile['name']}.")
    if user_profile.get("frequent_topics"):
        extras.append(f"They frequently ask about: {', '.join(user_profile['frequent_topics'])}.")
    if user_profile.get("notes"):
        extras.append(f"Notes: {'; '.join(user_profile['notes'][:3])}.")

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
    workflow.add_edge("decline", "profile_update")
    workflow.add_edge("profile_update", END)

    compiled = workflow.compile(checkpointer=checkpointer)
    # Set recursion limit once here — applies to CLI, Studio, MCP, everything.
    return compiled.with_config({"recursion_limit": _RECURSION_LIMIT})


# Module-level graph for langgraph dev / Studio (no checkpointer — Studio provides its own).
graph = build_graph()
