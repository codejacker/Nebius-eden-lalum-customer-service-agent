import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://api.studio.nebius.ai/v1/"

agent_llm = ChatOpenAI(
    model="NousResearch/Hermes-4-70B",
    base_url=_BASE_URL,
    api_key=os.environ["NEBIUS_API_KEY"],
    temperature=0,
)

router_llm = ChatOpenAI(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    base_url=_BASE_URL,
    api_key=os.environ["NEBIUS_API_KEY"],
    temperature=0,
)
