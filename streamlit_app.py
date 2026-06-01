"""
Streamlit chat UI for the Customer Service Data Analyst Agent.

Run:
    streamlit run streamlit_app.py
"""
import json

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

load_dotenv()

# Ensure the dataset exists before importing the agent (tools.py reads the CSV at import
# time). The CSV is gitignored, so on a fresh deploy (e.g. Streamlit Community Cloud) it is
# downloaded once on cold start, then cached on the ephemeral disk until the container restarts.
from pathlib import Path  # noqa: E402

if not Path("data/bitext.csv").exists():
    with st.spinner("Downloading dataset (one-time, ~20MB)…"):
        from download_data import download

        download()

from agent.graph import build_graph          # noqa: E402
from agent.memory import UserProfileManager, get_checkpointer  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Service Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached graph — built once, shared across all rerenders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_graph():
    return build_graph(checkpointer=get_checkpointer())


graph = get_graph()

# ---------------------------------------------------------------------------
# Sidebar — session management
# ---------------------------------------------------------------------------
def _existing_sessions() -> list[str]:
    """Return session names that have a saved profile, sorted alphabetically."""
    from pathlib import Path
    profiles_dir = Path("./profiles")
    if not profiles_dir.exists():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.json"))


with st.sidebar:
    st.title("Session")

    existing = _existing_sessions()
    options = existing + ["✚ New session…"]

    # Remember the active session across reruns. Without this, sending the first message
    # in a new session writes its profile file, which grows the options list and snaps the
    # (index-driven, key-less) selectbox back to the first option. Driving the index from
    # session_state keeps the selection put once the new session exists.
    if "session_id" not in st.session_state:
        st.session_state.session_id = existing[0] if existing else "default"

    active = st.session_state.session_id
    if active in existing:
        default_index = existing.index(active)
    else:
        default_index = len(options) - 1  # sit on "✚ New session…"

    selected = st.selectbox(
        "Choose session",
        options=options,
        index=default_index,
        help="Select an existing session to resume it, or create a new one.",
    )

    if selected == "✚ New session…":
        # key= persists the typed name across the rerun that processes the first message,
        # so the message is routed to the new session rather than falling back to default.
        session_input = st.text_input(
            "New session name",
            placeholder="e.g. alice",
            key="new_session_name",
            help="Leave blank for anonymous (no profile tracking).",
        )
        session_id = session_input.strip() or "default"
    else:
        session_id = selected

    st.session_state.session_id = session_id
    profile_enabled = session_id not in ("", "default")

    st.caption(
        "Profile tracking: **on**" if profile_enabled else "Profile tracking: **off** (use a named session to enable)"
    )

    # Show profile if one exists for this session
    if profile_enabled:
        with st.expander("User profile", expanded=True):
            profile = UserProfileManager(session_id).load()
            st.json(profile)

    st.divider()
    if st.button("Clear display history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session-switch guard — clear display history when session ID changes
# ---------------------------------------------------------------------------
if "active_session" not in st.session_state:
    st.session_state.active_session = session_id

if st.session_state.active_session != session_id:
    st.session_state.chat_history = []
    st.session_state.active_session = session_id

# ---------------------------------------------------------------------------
# Chat history initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📊 Customer Service Data Analyst")
st.caption(
    f"Session: **{session_id}** · Ask questions about the Bitext customer support dataset."
)

# Render existing chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant" and entry.get("steps"):
            with st.expander("Reasoning steps", expanded=False):
                for step in entry["steps"]:
                    st.markdown(step)
        st.markdown(entry["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about the dataset…")

if user_input:
    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    config = {
        "configurable": {
            "thread_id": session_id,
            "profile_enabled": profile_enabled,
        }
    }

    # Stream the agent response
    with st.chat_message("assistant"):
        steps = []
        final_answer = ""
        steps_placeholder = st.empty()
        answer_placeholder = st.empty()

        try:
            for event in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values",
            ):
                last_msg = event["messages"][-1]

                # Collect tool calls
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        args_str = json.dumps(tc["args"], ensure_ascii=False)
                        steps.append(f"**`[TOOL]`** `{tc['name']}({args_str})`")

                # Collect tool observations
                elif getattr(last_msg, "type", None) == "tool":
                    content = str(last_msg.content)
                    preview = content[:400] + "…" if len(content) > 400 else content
                    steps.append(f"**`[OBS]`**\n```\n{preview}\n```")

                # Final AI answer (no tool calls)
                elif getattr(last_msg, "type", None) == "ai" and not (
                    hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                ):
                    final_answer = last_msg.content or ""

                # Live-update reasoning steps while streaming
                if steps:
                    with steps_placeholder.expander("Reasoning steps", expanded=True):
                        for s in steps:
                            st.markdown(s)

            # Collapse reasoning once done and show final answer
            steps_placeholder.empty()
            if steps:
                with st.expander("Reasoning steps", expanded=False):
                    for s in steps:
                        st.markdown(s)

            answer_placeholder.markdown(final_answer or "_No response._")

        except GraphRecursionError:
            final_answer = (
                "I've reached my reasoning limit on this question. Try rephrasing."
            )
            answer_placeholder.markdown(final_answer)
        except Exception as e:
            final_answer = f"Error: {e}"
            answer_placeholder.markdown(final_answer)

    # Persist to display history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_answer, "steps": steps}
    )

    # The profile_update_node has just written the latest facts to disk during this run,
    # but the sidebar profile panel was rendered at the TOP of the script (before this
    # message was processed) — so it still shows the pre-message profile. Rerun once so
    # the sidebar reloads from disk and reflects the info from the message just sent.
    # chat_history is in session_state, so the conversation re-renders unchanged.
    if profile_enabled:
        st.rerun()
