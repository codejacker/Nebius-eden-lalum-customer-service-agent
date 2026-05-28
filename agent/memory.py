import json
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

_PROFILES_DIR = Path("./profiles")


# ---------------------------------------------------------------------------
# Task 2a — SqliteSaver checkpointer
# ---------------------------------------------------------------------------

def get_checkpointer() -> SqliteSaver:
    """Returns a SqliteSaver backed by memory.db in the project root.
    Uses check_same_thread=False so the CLI's single thread can reuse the connection."""
    conn = sqlite3.connect("./memory.db", check_same_thread=False)
    return SqliteSaver(conn)


# ---------------------------------------------------------------------------
# Task 2b — Per-user profile (separate from conversation history)
# ---------------------------------------------------------------------------

class UserProfileManager:
    """Manages per-user persistent profiles stored as JSON files in profiles/."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        _PROFILES_DIR.mkdir(exist_ok=True)
        self.path = _PROFILES_DIR / f"{session_id}.json"

    def load(self) -> dict:
        if self.path.exists():
            profile = json.loads(self.path.read_text())
            profile.pop("notes", None)  # migrate old profiles
            # Migrate frequent_topics from list → dict
            if isinstance(profile.get("frequent_topics"), list):
                profile["frequent_topics"] = {
                    t.upper(): 1 for t in profile["frequent_topics"]
                }
            return profile
        return {
            "name": None,
            "frequent_topics": {},   # {CATEGORY: mention_count}
            "preferences": {},
        }

    def save(self, profile: dict) -> None:
        self.path.write_text(json.dumps(profile, indent=2))

    def update_from_conversation(self, messages: list, llm) -> dict:
        """Ask the LLM to extract new user facts from the last 10 messages and merge them."""
        current = self.load()

        # Only look at human messages — the profile should reflect what the USER said,
        # not what the agent reported. AI responses mentioning categories would pollute
        # frequent_topics with inferred rather than stated interests.
        readable = [
            m for m in messages[-10:]
            if isinstance(m, HumanMessage) and m.content
        ]
        if not readable:
            return current  # nothing the user said — skip update
        conversation_text = "\n".join(
            f"user: {str(m.content)[:300]}" for m in readable
        )

        from agent.tools import CATEGORIES, INTENTS
        prompt = f"""You are updating a user profile based on a conversation with a data analyst agent.
The agent answers questions about a customer service dataset.

Current profile:
{json.dumps(current, indent=2)}

Recent conversation (user messages only):
{conversation_text}

Valid dataset categories: {CATEGORIES}
Valid dataset intents: {INTENTS}

Extract NEW facts about the user and return ONLY a JSON object with the fields that changed.
The profile has exactly three fields:

1. "name" — the user's name if they explicitly mentioned it. String or null.

2. "frequent_topics" — a dict mapping category name to 1 for each category the user showed
   interest in during this conversation. Add a category when:
   • The user expressed interest in it: "i like refunds", "i care about cancellations"
   • The user queried about it: "show me examples from SHIPPING", "how many ORDER rows"
   • The user named it or a clear synonym: "refunds"→REFUND, "cancellations"→CANCEL,
     "payments"→PAYMENT, "shipping"→SHIPPING, "orders"→ORDER, "account"→ACCOUNT
   RULES:
   - Key MUST be an exact value from: {CATEGORIES}
   - Do NOT infer from unrelated words — "money" is NOT PAYMENT or INVOICE
   - Return as a dict: {{"REFUND": 1, "SHIPPING": 1}} not a list

3. "preferences" — a flat key-value dict capturing how the user wants to interact or what
   they are focused on. A single statement can populate BOTH frequent_topics AND preferences.
   Examples:
   • "i like refunds and cancellations oriented questions"
     → frequent_topics: {{"REFUND": 1, "CANCEL": 1}}, preferences: {{"focus": "refunds and cancellations"}}
   • "i prefer concise answers" → preferences: {{"output": "concise"}}
   • "show me examples" → preferences: {{"style": "examples"}}
   Only add what the user clearly stated or strongly implied. Do NOT infer or generalise.

Rules:
- Return ONLY valid JSON — no explanation, no markdown fences.
- Only include fields that changed. Omit unchanged fields.
- If nothing new was learned, return {{}}.
- Do not repeat items already in the current profile."""

        response = llm.invoke(prompt)

        try:
            content = response.content.strip()
            # Strip markdown fences if the LLM wraps its JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            updates = json.loads(content)

            _VALID_KEYS = {"name", "frequent_topics", "preferences"}
            for key, value in updates.items():
                if key not in _VALID_KEYS:
                    continue  # ignore unknown fields (e.g. old "notes")
                if key == "frequent_topics" and isinstance(value, dict):
                    from agent.tools import CATEGORIES
                    topics = current.setdefault("frequent_topics", {})
                    for cat, count in value.items():
                        cat_upper = cat.upper()
                        if cat_upper in CATEGORIES:  # hard validate against loaded schema
                            topics[cat_upper] = topics.get(cat_upper, 0) + int(count)
                elif key == "preferences" and isinstance(value, dict):
                    current.setdefault("preferences", {}).update(
                        {k: v for k, v in value.items() if v}
                    )
                elif value:  # name — don't overwrite with null/empty
                    current[key] = value

            self.save(current)
        except (json.JSONDecodeError, IndexError, ValueError):
            pass  # malformed LLM output — skip silently, profile unchanged

        return current
