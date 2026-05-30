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

        # Process ONLY the latest human message. profile_update_node runs once per user
        # turn, so this counts each message EXACTLY once. Reading a sliding window of the
        # last N messages re-counted the same message on every later turn — inflating
        # counts (e.g. REFUND climbing on a turn that was about ACCOUNT) and defeating the
        # ≥2 "frequent" threshold. We also look at human messages only — the profile must
        # reflect what the USER said, not categories the agent happened to list.
        human_msgs = [m for m in messages if isinstance(m, HumanMessage) and m.content]
        if not human_msgs:
            return current  # nothing the user said — skip update
        latest = human_msgs[-1]
        conversation_text = f"user: {str(latest.content)[:300]}"

        from agent.tools import CATEGORIES, INTENTS
        prompt = f"""You are updating a user profile based on a conversation with a data analyst agent.
The agent answers questions about a customer service dataset.

Current profile:
{json.dumps(current, indent=2)}

User's latest message:
{conversation_text}

Valid dataset categories: {CATEGORIES}
Valid dataset intents: {INTENTS}

Extract NEW facts about the user and return ONLY a JSON object with the fields that changed.
The profile has exactly three fields:

1. "name" — the user's name if they explicitly mentioned it. String or null.

2. "frequent_topics" — a dict mapping category name to 1 for each SPECIFIC category the
   user named or asked about in THIS message. Add a category ONLY when:
   • The user expressed interest in it: "i like refunds", "i care about cancellations"
   • The user queried that specific category: "show me examples from SHIPPING", "how many ORDER rows"
   • The user named it or a clear synonym: "refunds"→REFUND, "cancellations"→CANCEL,
     "payments"→PAYMENT, "shipping"→SHIPPING, "orders"→ORDER, "account"→ACCOUNT
   DO NOT add a category when:
   • The user asks what categories EXIST, or to LIST/enumerate them
     ("what categories are there?", "list all categories"). That is general curiosity
     about the schema — NOT interest in any specific category. Return {{}} for topics.
   • A category name is not actually present in the user's own words.
   RULES:
   - Key MUST be an exact value from: {CATEGORIES}
   - Do NOT infer from unrelated words — "money" is NOT PAYMENT or INVOICE
   - ALWAYS return a category the user named here even if it is already in the current
     profile — its mention count is incremented on merge (this drives the "frequent" threshold).
   - Return as a dict: {{"REFUND": 1}} not a list

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
- For "name" and "preferences": do NOT repeat a value already in the current profile.
- For "frequent_topics": ALWAYS include any category the user named in this message,
  even if it already appears in the profile — the count must be incremented."""

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
