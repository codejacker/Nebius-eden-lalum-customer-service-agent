from agent.llms import router_llm
from agent.state import AgentState
from agent.tools import CATEGORIES, INTENTS

_ROUTER_PROMPT = """You are a query classifier for a customer service data analyst agent.
The agent can only answer questions about the Bitext Customer Service dataset.

Dataset covers these categories: {categories}
Dataset covers these intents: {intents}

{context}Classify the user's question as exactly one of:
- "structured"   : has a concrete data-driven answer (counts, lists, examples, distributions).
                   Also use this for follow-up requests ("show me more", "3 more", "what about X")
                   when context shows the conversation is already about the dataset.
- "unstructured" : requires summarisation or qualitative analysis of the data.
                   ALSO use when the user provides personal context about themselves, including:
                   • Their name — e.g. "my name is eden", "i'm eden", "call me eden", "name eden"
                   • An interest in a dataset topic — e.g. "i like refunds", "i'm into shipping"
                   • An interaction preference — e.g. "i prefer examples", "keep it concise"
                   • Asking what the agent knows about them — e.g. "what do you remember about me?"
- "out_of_scope" : ONLY for topics with no connection to the agent or dataset at all —
                   weather, politics, sports, coding, general trivia.
                   Do NOT use if the message mentions a dataset topic or is about the user themselves.

Return ONLY the single classification word. Nothing else.

Question: {question}"""


def router_node(state: AgentState) -> AgentState:
    # Empty or missing message — nothing to work with, decline cleanly.
    if not state.get("messages"):
        return {"query_type": "out_of_scope"}
    last = state["messages"][-1]
    if not getattr(last, "content", "").strip():
        return {"query_type": "out_of_scope"}

    question = last.content

    # Pass up to the last 4 messages as context so the router understands follow-ups
    prior = state["messages"][:-1][-4:]
    if prior:
        lines = "\n".join(f"  {m.type}: {str(m.content)[:120]}" for m in prior)
        context = f"Recent conversation:\n{lines}\n\n"
    else:
        context = ""

    prompt = _ROUTER_PROMPT.format(
        categories=CATEGORIES,
        intents=INTENTS,
        context=context,
        question=question,
    )
    response = router_llm.invoke(prompt)
    classification = response.content.strip().lower()

    if classification not in ("structured", "unstructured", "out_of_scope"):
        classification = "structured"

    return {"query_type": classification}
