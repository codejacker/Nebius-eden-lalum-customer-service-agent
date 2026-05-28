from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.llms import agent_llm

# ---------------------------------------------------------------------------
# Dataset — loaded once at import time. 19MB fits comfortably in memory.
# ---------------------------------------------------------------------------
_DATA_PATH = Path(__file__).parent.parent / "data" / "bitext.csv"
df = pd.read_csv(_DATA_PATH)
df["category"] = df["category"].str.upper()
df["intent"] = df["intent"].str.lower()

CATEGORIES: list[str] = sorted(df["category"].unique().tolist())
INTENTS: list[str] = sorted(df["intent"].unique().tolist())

# Computed once at load — used by graph.py for a dynamic (never stale) system prompt line.
DATASET_INFO: dict = {
    "total_rows": len(df),
    "columns": df.columns.tolist(),
    "num_categories": df["category"].nunique(),
    "num_intents": df["intent"].nunique(),
}


# ---------------------------------------------------------------------------
# Tool 1 — schema overview
# ---------------------------------------------------------------------------
@tool
def get_dataset_schema() -> dict:
    """Returns the Bitext dataset schema: column names, all categories, all intents, and total row count.
    Call this first if you are unsure what categories or intents exist."""
    return {
        "columns": df.columns.tolist(),
        "total_rows": len(df),
        "categories": CATEGORIES,
        "intents": INTENTS,
    }


# ---------------------------------------------------------------------------
# Tool 2 — list categories
# ---------------------------------------------------------------------------
@tool
def list_categories() -> list[str]:
    """Lists all 10 available categories in the Bitext customer service dataset
    (e.g. REFUND, ACCOUNT, ORDER). Use when the user asks what topics are covered."""
    return CATEGORIES


# ---------------------------------------------------------------------------
# Tool 3 — list intents within a category
# ---------------------------------------------------------------------------
class ListIntentsInput(BaseModel):
    category: str = Field(description="Category name, e.g. 'REFUND' or 'ACCOUNT'. Case-insensitive.")


@tool(args_schema=ListIntentsInput)
def list_intents(category: str) -> list[str]:
    """Lists all intents within a given category.
    Use this to discover exact intent names before filtering by intent."""
    filtered = df[df["category"] == category.upper()]
    if filtered.empty:
        return [f"No category '{category}' found. Available categories: {CATEGORIES}"]
    return sorted(filtered["intent"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 4 — count rows
# ---------------------------------------------------------------------------
class CountRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter by category (e.g. 'REFUND'). Omit to count all rows.",
    )
    intent: str | None = Field(
        default=None,
        description="Filter by intent (e.g. 'track_refund'). Omit to count all rows.",
    )


@tool(args_schema=CountRowsInput)
def count_rows(category: str | None = None, intent: str | None = None) -> dict:
    """Returns the number of rows in the Bitext dataset, optionally filtered by category and/or intent.
    Returns an error dict if the category or intent does not exist in the dataset."""
    result = df
    if category:
        cat_upper = category.upper()
        if cat_upper not in CATEGORIES:
            return {"error": f"Category '{category}' not found. Available categories: {CATEGORIES}"}
        result = result[result["category"] == cat_upper]
    if intent:
        intent_lower = intent.lower()
        if intent_lower not in INTENTS:
            return {"error": f"Intent '{intent}' not found. Call list_intents(category) to see valid intents."}
        result = result[result["intent"] == intent_lower]
    return {"count": len(result)}


# ---------------------------------------------------------------------------
# Tool 5 — get example rows
# ---------------------------------------------------------------------------
class GetExamplesInput(BaseModel):
    n: int = Field(default=5, ge=1, le=50, description="Number of examples to return (1–50).")
    category: str | None = Field(
        default=None,
        description="Filter by category (e.g. 'REFUND'). Case-insensitive.",
    )
    intent: str | None = Field(
        default=None,
        description="Filter by intent (e.g. 'track_refund'). Case-insensitive.",
    )
    keyword: str | None = Field(
        default=None,
        description=(
            "Search term within customer instruction text. "
            "Use when the user describes a topic in plain language instead of naming a category or intent. "
            "Example: 'money back', 'cancel', 'lost package'."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many rows before returning results. Use for follow-up 'show me more' requests.",
    )


@tool(args_schema=GetExamplesInput)
def get_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """Returns example rows from the Bitext dataset as a list of dicts.
    Each dict has keys: category, intent, instruction (customer message), response (agent reply).
    Use keyword search when the user describes a topic without naming a specific category or intent."""
    result = df
    if category:
        cat_upper = category.upper()
        if cat_upper not in CATEGORIES:
            return [{"error": f"Category '{category}' not found. Available categories: {CATEGORIES}"}]
        result = result[result["category"] == cat_upper]
    if intent:
        intent_lower = intent.lower()
        if intent_lower not in INTENTS:
            return [{"error": f"Intent '{intent}' not found. Call list_intents(category) to see valid intents."}]
        result = result[result["intent"] == intent_lower]
    if keyword:
        result = result[result["instruction"].str.contains(keyword, case=False, na=False)]

    if result.empty:
        return [{"error": "No rows matched the given filters."}]

    result = result.iloc[offset : offset + n]
    return result[["category", "intent", "instruction", "response"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Tool 6 — intent distribution within a category
# ---------------------------------------------------------------------------
class GetIntentDistributionInput(BaseModel):
    category: str = Field(
        description="The category to analyse (e.g. 'ACCOUNT'). Case-insensitive."
    )


@tool(args_schema=GetIntentDistributionInput)
def get_intent_distribution(category: str) -> dict:
    """Returns the row count for each intent within a given category.
    Use this to show a breakdown / distribution of sub-topics."""
    filtered = df[df["category"] == category.upper()]
    if filtered.empty:
        return {"error": f"No category '{category}' found. Available: {CATEGORIES}"}
    return filtered.groupby("intent").size().sort_values(ascending=False).to_dict()


# ---------------------------------------------------------------------------
# Tool 7 — qualitative summary of agent responses
# ---------------------------------------------------------------------------
class SummarizeResponsesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter by category before summarising.",
    )
    intent: str | None = Field(
        default=None,
        description="Filter by intent before summarising.",
    )


@tool(args_schema=SummarizeResponsesInput)
def summarize_responses(
    category: str | None = None,
    intent: str | None = None,
) -> str:
    """Generates a qualitative LLM summary of how customer service agents respond for a given
    category or intent. Use for 'how do agents handle X' or 'what tone do responses use' questions."""
    result = df
    if category:
        result = result[result["category"] == category.upper()]
    if intent:
        result = result[result["intent"] == intent.lower()]

    if result.empty:
        return "No rows matched the given filters."

    sample = result["response"].sample(min(20, len(result)), random_state=42).tolist()
    responses_text = "\n---\n".join(sample)

    label = f"intent '{intent}'" if intent else f"category '{category}'"
    prompt = f"""You are analysing customer service responses from the Bitext dataset.
Below are {len(sample)} example agent responses for {label}.

{responses_text}

Write a concise 2–3 paragraph summary covering:
1. The main themes in how agents respond
2. The tone and style used
3. Any notable patterns or approaches"""

    response = agent_llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# Exported list — used by graph.py to bind tools to the LLM
# ---------------------------------------------------------------------------
TOOLS = [
    get_dataset_schema,
    list_categories,
    list_intents,
    count_rows,
    get_examples,
    get_intent_distribution,
    summarize_responses,
]
