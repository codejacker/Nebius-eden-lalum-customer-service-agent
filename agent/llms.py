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

# Personal/meta turns produce prose, not tool calls. Hermes is tool-fine-tuned and
# overrides prose instructions to emit function-call syntax even with no tools bound.
# A peer-grade INSTRUCTION-tuned model follows the template instead of fighting it.
personal_llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    base_url=_BASE_URL,
    api_key=os.environ["NEBIUS_API_KEY"],
    temperature=0,
)
