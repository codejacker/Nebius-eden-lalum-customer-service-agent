from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: NotRequired[str]
    user_profile: NotRequired[dict]
