from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llms import agent_llm, personal_llm
from agent.memory import UserProfileManager
from agent.router import router_node
from agent.state import AgentState
from agent.tools import CATEGORIES, TOOLS

_RECURSION_LIMIT = 12
_MAX_MESSAGES = 20


def _get_system_prompt(user_profile: dict) -> str:
    base = """You are a data analyst assistant with tools to query a structured tabular dataset.

Rules you must follow:
1. If you are unsure of the dataset's structure, call get_dataset_schema() first — it returns columns, row count, and available filter values.
2. NEVER assume what categories or intents exist. Verify a category exists before filtering by it. To count ALL rows in a category call count_rows(category=...) directly — do NOT call list_intents() first. Only call list_intents() when you need to filter by a specific intent.
3. Always use your tools for data questions — never guess at counts, names, or distributions.
4. When asked to total or sum counts from earlier in this conversation, add the exact numbers that tools already returned in this session. Never invent, guess, or round. Cross-session aggregation is not supported.
5. When the user asks to SEE examples, reproduce each row VERBATIM — never summarise, shorten, or paraphrase. Copy the `instruction` (customer message) and `response` (agent reply) fields exactly as the tool returned them, including any {{placeholder}} tokens. Render each example using this exact Markdown layout, with a blank line between every block so the customer and agent text are visually separated:

   **Example 1** — _category / intent_

   **Customer:** <verbatim instruction text>

   **Agent:** <verbatim response text>

   (then a blank line, then "Example 2", and so on.)

   This rule OVERRIDES any conciseness preference — example fidelity and clear separation matter more than brevity. (This does not apply to summary requests, which use summarize_responses.)
6. Be concise and factual — EXCEPT when showing examples (rule 5), which must stay verbatim.
7. If a question is clearly unrelated to the dataset, politely decline."""
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


def _get_personal_prompt(user_profile: dict) -> str:
    profile_lines = []
    if user_profile.get("name"):
        profile_lines.append(f"Name: {user_profile['name']}")
    raw_topics = user_profile.get("frequent_topics", {})
    if isinstance(raw_topics, dict):
        frequent = [f"{t} ({c}x)" for t, c in raw_topics.items() if c >= 2]
        any_topics = [t for t in raw_topics.keys()]
    else:
        frequent = list(raw_topics)
        any_topics = frequent
    if frequent:
        profile_lines.append(f"Frequent topics (mentioned 2+ times): {', '.join(frequent)}")
    elif any_topics:
        profile_lines.append(f"Topics mentioned so far: {', '.join(any_topics)}")
    if user_profile.get("preferences"):
        prefs = "; ".join(f"{k}: {v}" for k, v in user_profile["preferences"].items())
        profile_lines.append(f"Preferences: {prefs}")

    profile_text = "\n".join(profile_lines) if profile_lines else "No information recorded yet."

    return f"""You are a conversational assistant helping a user explore a customer service dataset.
The user has just sent a personal or meta message — not a data query. Do NOT call any tools.

User profile (what you know about them):
{profile_text}

All categories that exist in the dataset: {CATEGORIES}

Choose exactly one of these three response modes based on what the user said:

A) ACKNOWLEDGE — user stated their name, a preference, or an interest. This includes
   interests unrelated to the dataset ("I like money", "I like dogs"):
   Respond warmly in 1-2 sentences confirming what you heard. Do not turn it into a query.
   If the interest is outside the dataset, acknowledge it politely without forcing a
   dataset connection ("Nice! Dogs aren't something I track, but I'm here for any
   customer-service question.").
   Example: "Got it, I'll remember that you prefer concise answers."

B) SUMMARIZE — user asked what you know or remember about them:
   Tell them exactly what is in the profile above. If empty, say so honestly.

C) SUGGEST — user asked what to query next, wants a suggestion, or asks which category to explore:
   - For interest-based suggestions, use a topic that ACTUALLY appears in the profile above.
     NEVER claim the user is interested in a category that is not in their profile (e.g. do
     not say "based on your interest in PAYMENT" if PAYMENT is not listed above).
   - If the user wants a DIFFERENT category than usual: either pick another topic that IS in
     their profile (e.g. their 2nd-highest), or pick one from the category list above and be
     HONEST that it is a new area they have not explored — do not pretend it was a past interest.
   - Factor in their stated style preference (e.g. style: examples → suggest seeing examples).
   Make ONE specific suggestion in fluent plain English (e.g. "see 5 examples from the CANCEL
   category"), not a vague "examples or statistics". NEVER write function names, parameters, or code.
   Use this two-sentence pattern (adapt the opener if it is a new/unexplored category):
   "Based on your interest in [topic from profile], you might want to [specific plain English]. Should I go ahead?"
   Do not execute anything — just suggest and ask for confirmation.

Respond in plain conversational prose. No tools, no code."""


def _conversational_history(messages: list) -> list:
    """Keep only human turns and prose AI replies for the personal node.

    The personal node must never see prior tool calls — Hermes is tool-fine-tuned and
    will mimic any function-call syntax it finds in the transcript, leaking it into
    suggestions. We drop tool-call AIMessages and ToolMessages AS A PAIR (never strip a
    result while keeping its call, which would orphan the tool_call and break the API).
    Prose AIMessages (no tool_calls) are kept so refinement turns still have context."""
    return [
        m
        for m in messages
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content)
    ]


_agent_with_tools = agent_llm.bind_tools(TOOLS)


def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    session_id = config.get("configurable", {}).get("thread_id", "default")
    profile = state.get("user_profile") or UserProfileManager(session_id).load()
    messages = state["messages"][-_MAX_MESSAGES:]
    response = _agent_with_tools.invoke(
        [SystemMessage(content=_get_system_prompt(profile))] + messages
    )
    return {"messages": [response]}


def personal_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handles personal/meta messages (name, preferences, profile summary, suggestions).
    Uses personal_llm (instruction-tuned, no tools) — Hermes overrides the prose template
    with function-call syntax even unbound, so a non-tool-tuned model answers these turns."""
    session_id = config.get("configurable", {}).get("thread_id", "default")
    profile = state.get("user_profile") or UserProfileManager(session_id).load()
    # Strip tool-call/tool-result messages so there is no function-call syntax to mimic.
    messages = _conversational_history(state["messages"])[-_MAX_MESSAGES:]
    response = personal_llm.invoke(
        [SystemMessage(content=_get_personal_prompt(profile))] + messages
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
    workflow.add_node("personal", personal_node)
    workflow.add_node("profile_update", profile_update_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s["query_type"],
        {
            "structured": "agent",
            "unstructured": "agent",
            "personal": "personal",
            "out_of_scope": "decline",
        },
    )
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "profile_update"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("personal", "profile_update")
    workflow.add_edge("decline", END)
    workflow.add_edge("profile_update", END)

    compiled = workflow.compile(checkpointer=checkpointer)
    # Set recursion limit once here — applies to CLI, Studio, MCP, everything.
    return compiled.with_config({"recursion_limit": _RECURSION_LIMIT})


# Module-level graph for langgraph dev / Studio (no checkpointer — Studio provides its own).
graph = build_graph()
