from agent.llms import router_llm
from agent.state import AgentState
from agent.tools import CATEGORIES, INTENTS

_ROUTER_PROMPT = """You are a query classifier for a customer service data analyst agent.
The agent can only answer questions about the Bitext Customer Service dataset.

Dataset covers these categories: {categories}
Dataset covers these intents: {intents}

{context}Classify the user's question as exactly one of:
- "structured"   : asks for data — counts, lists, examples, distributions, comparisons.
                   Use this for follow-ups ("show me more", "3 more", "what about X") when
                   context shows the conversation is already about the dataset.
                   Use this for ANY message that contains a data ask, even if it also
                   mentions the user's name or preferences.
                   ALSO use this for a short CONFIRMATION ("yes", "sure", "ok", "go ahead",
                   "do it", "yes please") WHEN the recent conversation shows the assistant
                   just offered a query suggestion and asked for confirmation — the user is
                   approving it, so it must now be executed with tools.
- "unstructured" : requires qualitative analysis or summarisation of the dataset
                   (e.g. "summarise how agents handle refunds", "describe the tone").
                   Use this for ANY message that contains such an analysis request, even
                   if it also mentions the user's name or preferences.
- "personal"     : the ENTIRE message is about the user or the conversation itself —
                   no data to look up, no analysis to do. Examples:
                   • User states their name — "my name is eden", "call me eden"
                   • User states a preference or interest — "i like refunds", "i prefer examples"
                   • User asks what you know about them — "what do you remember about me?"
                   • User asks what to query next — "what should I ask?", "suggest something"
                   If the message also asks for any count, list, example, or distribution,
                   do NOT use personal — use structured or unstructured instead.
- "out_of_scope" : the topic has no connection to the agent or dataset at all —
                   weather, politics, sports, coding, general trivia.

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

    if classification not in ("structured", "unstructured", "personal", "out_of_scope"):
        classification = "structured"

    return {"query_type": classification}
