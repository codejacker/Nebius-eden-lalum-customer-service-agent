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
            return json.loads(self.path.read_text())
        return {
            "name": None,
            "frequent_topics": [],
            "preferences": {},
            "notes": [],
        }

    def save(self, profile: dict) -> None:
        self.path.write_text(json.dumps(profile, indent=2))

    def update_from_conversation(self, messages: list, llm) -> dict:
        """Ask the LLM to extract new user facts from the last 10 messages and merge them."""
        current = self.load()

        # Only look at human and AI messages — tool messages are noisy data
        readable = [
            m for m in messages[-10:]
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
        conversation_text = "\n".join(
            f"{m.type}: {str(m.content)[:300]}" for m in readable
        )

        prompt = f"""You are updating a user profile based on a conversation with a data analyst agent.

Current profile:
{json.dumps(current, indent=2)}

Recent conversation:
{conversation_text}

Extract any NEW facts about the user (e.g. their name, topics they care about, preferences they expressed).
If the user mentioned their name, add it to "name".
If the user showed interest in a topic (REFUND, ACCOUNT, etc.), add it to "frequent_topics".
Add any other notable preferences or observations to "notes".

Rules:
- Return ONLY valid JSON — no explanation, no markdown fences.
- Only include fields that changed or were added.
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

            for key, value in updates.items():
                if isinstance(current.get(key), list) and isinstance(value, list):
                    # Append only genuinely new items to lists
                    existing = set(str(v).lower() for v in current[key])
                    current[key] += [v for v in value if str(v).lower() not in existing]
                elif value:  # don't overwrite with null/empty
                    current[key] = value

            self.save(current)
        except (json.JSONDecodeError, IndexError, ValueError):
            pass  # malformed LLM output — skip silently, profile unchanged

        return current
