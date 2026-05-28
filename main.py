"""
CLI entry point for the Customer Service Data Analyst agent.

Usage:
    python main.py                          # default session, persisted to memory.db
    python main.py --session alice          # named session
    python main.py --session alice --show-profile   # print alice's profile and exit
    python main.py --list-sessions          # list all known sessions and exit
    python main.py --session alice --no-persist     # RAM-only, nothing written to disk
"""
import argparse
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

load_dotenv()

from agent.graph import build_graph          # noqa: E402
from agent.memory import UserProfileManager, get_checkpointer  # noqa: E402


def cmd_show_profile(session_id: str) -> None:
    profile = UserProfileManager(session_id).load()
    print(f"\nProfile for session '{session_id}':")
    print(json.dumps(profile, indent=2))


def cmd_list_sessions() -> None:
    from pathlib import Path
    profiles_dir = Path("./profiles")
    files = sorted(profiles_dir.glob("*.json")) if profiles_dir.exists() else []
    if not files:
        print("No sessions found. Start one with: python main.py --session <name>")
        return
    print(f"\n{'Session':<25} {'Name':<15} Topics")
    print("-" * 60)
    for f in files:
        sid = f.stem
        try:
            p = json.loads(f.read_text())
            name = p.get("name") or "—"
            topics = ", ".join(p.get("frequent_topics", [])) or "—"
        except Exception:
            name, topics = "?", "?"
        print(f"{sid:<25} {name:<15} {topics}")


def main():
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst")
    parser.add_argument("--session", default=None,
                        help="Session ID — determines which conversation and profile to load. "
                             "Omitting this disables profile tracking.")
    parser.add_argument("--no-persist", action="store_true",
                        help="Use in-memory storage only — nothing written to memory.db")
    parser.add_argument("--show-profile", action="store_true",
                        help="Print the profile for --session and exit")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List all sessions with saved profiles and exit")
    args = parser.parse_args()

    # ── Info commands (exit after printing) ──────────────────────────────────
    if args.list_sessions:
        cmd_list_sessions()
        return
    if args.show_profile:
        cmd_show_profile(args.session)
        return

    # ── Build graph ──────────────────────────────────────────────────────────
    if args.no_persist:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        persist_label = "RAM only (no-persist)"
    else:
        checkpointer = get_checkpointer()
        persist_label = "memory.db"

    session_id = args.session or "default"
    profile_enabled = args.session is not None  # only track profile for named sessions

    graph = build_graph(checkpointer=checkpointer)
    # recursion_limit is set inside build_graph via .with_config() — not needed here.
    config = {
        "configurable": {
            "thread_id": session_id,
            "profile_enabled": profile_enabled,
        }
    }

    session_label = args.session if args.session else "default (no profile tracking)"
    print("\n=== Customer Service Data Analyst ===")
    print(f"Session: {session_label}  |  Storage: {persist_label}  |  Type 'quit' to exit")
    print("=" * 45 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            final_event = None
            for event in graph.stream(
                {"messages": [HumanMessage(content=query)]},
                config,
                stream_mode="values",
            ):
                last_msg = event["messages"][-1]

                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"  [TOOL] {tc['name']}({tc['args']})")
                elif getattr(last_msg, "type", None) == "tool":
                    content = str(last_msg.content)
                    preview = content[:400] + "…" if len(content) > 400 else content
                    print(f"  [OBS]  {preview}")

                final_event = event

            if final_event:
                print(f"\nAgent: {final_event['messages'][-1].content}\n")

        except GraphRecursionError:
            print("Agent: I've reached my reasoning limit on this question. Try rephrasing.\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
