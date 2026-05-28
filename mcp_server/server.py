"""
FastMCP server — exposes Bitext dataset tools over the MCP protocol.

Run:
    python mcp_server/server.py

Connect (Python client example):
    from fastmcp import Client
    import asyncio

    async def main():
        async with Client("python mcp_server/server.py") as client:
            print(await client.call_tool("list_categories", {}))

    asyncio.run(main())
"""
from fastmcp import FastMCP

# Re-use the already-loaded DataFrame and validated lists from the agent package.
# The dataset is loaded once at import time in agent/tools.py — we share that
# in-memory copy rather than loading it a second time.
from agent.tools import df, CATEGORIES, INTENTS

mcp = FastMCP(
    name="Bitext Customer Service Analyst",
    instructions=(
        "Tools for querying the Bitext Customer Support dataset. "
        "The dataset contains 26,872 labelled customer-service exchanges "
        "across 11 categories and 27 intents. "
        "Call list_categories first if you are unsure what categories exist."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 — list categories
# ---------------------------------------------------------------------------
@mcp.tool
def list_categories() -> list[str]:
    """Return all top-level categories in the Bitext dataset
    (e.g. REFUND, ACCOUNT, ORDER). Use this before filtering by category."""
    return CATEGORIES


# ---------------------------------------------------------------------------
# Tool 2 — list intents within a category
# ---------------------------------------------------------------------------
@mcp.tool
def list_intents(category: str) -> list[str]:
    """Return all intent names within a given category.
    Call this before filtering by intent to get the exact names.

    Args:
        category: Category name, e.g. 'REFUND' or 'ACCOUNT'. Case-insensitive.
    """
    cat_upper = category.upper()
    if cat_upper not in CATEGORIES:
        return [f"Category '{category}' not found. Available: {CATEGORIES}"]
    return sorted(df[df["category"] == cat_upper]["intent"].unique().tolist())


# ---------------------------------------------------------------------------
# Tool 3 — count rows
# ---------------------------------------------------------------------------
@mcp.tool
def count_rows(category: str | None = None, intent: str | None = None) -> dict:
    """Return the number of rows in the dataset, optionally filtered.

    Args:
        category: Optional. Filter by category (e.g. 'REFUND'). Case-insensitive.
        intent:   Optional. Filter by intent (e.g. 'track_refund'). Case-insensitive.
    """
    result = df
    if category:
        cat_upper = category.upper()
        if cat_upper not in CATEGORIES:
            return {"error": f"Category '{category}' not found. Available: {CATEGORIES}"}
        result = result[result["category"] == cat_upper]
    if intent:
        intent_lower = intent.lower()
        if intent_lower not in INTENTS:
            return {"error": f"Intent '{intent}' not found. Call list_intents(category) first."}
        result = result[result["intent"] == intent_lower]
    return {"count": len(result)}


# ---------------------------------------------------------------------------
# Tool 4 — get example rows
# ---------------------------------------------------------------------------
@mcp.tool
def get_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """Return example rows from the dataset as a list of dicts.
    Each dict has keys: category, intent, instruction, response.

    Args:
        n:        Number of examples to return (1–50). Default 5.
        category: Optional. Filter by category (e.g. 'REFUND'). Case-insensitive.
        intent:   Optional. Filter by intent (e.g. 'track_refund'). Case-insensitive.
        keyword:  Optional. Search term within customer instruction text.
        offset:   Skip this many rows before returning. Use for pagination.
    """
    n = max(1, min(n, 50))
    result = df

    if category:
        cat_upper = category.upper()
        if cat_upper not in CATEGORIES:
            return [{"error": f"Category '{category}' not found. Available: {CATEGORIES}"}]
        result = result[result["category"] == cat_upper]
    if intent:
        intent_lower = intent.lower()
        if intent_lower not in INTENTS:
            return [{"error": f"Intent '{intent}' not found. Call list_intents(category) first."}]
        result = result[result["intent"] == intent_lower]
    if keyword:
        result = result[result["instruction"].str.contains(keyword, case=False, na=False)]

    if result.empty:
        return [{"error": "No rows matched the given filters."}]

    return result.iloc[offset: offset + n][
        ["category", "intent", "instruction", "response"]
    ].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Tool 5 — intent distribution within a category
# ---------------------------------------------------------------------------
@mcp.tool
def get_intent_distribution(category: str) -> dict:
    """Return the row count per intent for a given category.
    Useful for understanding the breakdown of sub-topics.

    Args:
        category: Category to analyse (e.g. 'ACCOUNT'). Case-insensitive.
    """
    cat_upper = category.upper()
    if cat_upper not in CATEGORIES:
        return {"error": f"Category '{category}' not found. Available: {CATEGORIES}"}
    filtered = df[df["category"] == cat_upper]
    return filtered.groupby("intent").size().sort_values(ascending=False).to_dict()


# ---------------------------------------------------------------------------
# Entry point — stdio transport (default, works with any MCP client)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
