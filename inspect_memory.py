"""
Inspect memory.db — view sessions, message history, and user profiles.

Usage:
    python inspect_memory.py --sessions                 # list all thread IDs in memory.db
    python inspect_memory.py --messages alice           # print recent messages for a session
    python inspect_memory.py --messages alice --n 20    # print last 20 messages
    python inspect_memory.py --profile alice            # print alice's profile JSON
"""
import argparse
import json

from dotenv import load_dotenv

load_dotenv()


def cmd_sessions() -> None:
    from agent.memory import get_checkpointer
    checkpointer = get_checkpointer()

    # LangGraph's SqliteSaver exposes the raw sqlite connection
    conn = checkpointer.conn
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    except Exception:
        # Newer schema uses 'checkpoints' differently — try fallback
        try:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
        except Exception as e:
            print(f"Could not read memory.db: {e}")
            return

    if not rows:
        print("memory.db exists but has no sessions yet.")
        return

    print(f"\n{len(rows)} session(s) in memory.db:")
    for (tid,) in rows:
        print(f"  • {tid}")


def cmd_messages(session_id: str, n: int) -> None:
    from agent.memory import get_checkpointer
    checkpointer = get_checkpointer()

    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if checkpoint_tuple is None:
        print(f"No session '{session_id}' found in memory.db.")
        return

    messages = (
        checkpoint_tuple.checkpoint
        .get("channel_values", {})
        .get("messages", [])
    )

    if not messages:
        print(f"Session '{session_id}' exists but has no messages yet.")
        return

    recent = messages[-n:]
    print(f"\nLast {len(recent)} of {len(messages)} messages — session '{session_id}':")
    print("─" * 60)
    for msg in recent:
        msg_type = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        content_str = str(content)
        preview = content_str[:300] + "…" if len(content_str) > 300 else content_str
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            calls = ", ".join(tc["name"] for tc in tool_calls)
            print(f"[{msg_type}] → tool calls: {calls}")
        else:
            print(f"[{msg_type}] {preview}")
        print()


def cmd_profile(session_id: str) -> None:
    from agent.memory import UserProfileManager
    profile = UserProfileManager(session_id).load()
    print(f"\nProfile — '{session_id}':")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect agent memory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sessions", action="store_true",
                       help="List all thread IDs in memory.db")
    group.add_argument("--messages", metavar="SESSION_ID",
                       help="Print recent messages for a session")
    group.add_argument("--profile", metavar="SESSION_ID",
                       help="Print the user profile JSON for a session")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of messages to show (default 10)")
    args = parser.parse_args()

    if args.sessions:
        cmd_sessions()
    elif args.messages:
        cmd_messages(args.messages, args.n)
    elif args.profile:
        cmd_profile(args.profile)
