from langchain_openai import ChatOpenAI
from VoiceState import VoiceState

# Create the LLM instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=500,
)
TURN_LIMIT = 4

async def llm_service(state: VoiceState):  # ✅ async def
    """
    Async LLM service node for LangGraph workflow
    """
    messages = state["messages"]
    trimmed_messages = messages[-TURN_LIMIT * 2:]

    system_prompt = {
        "role": "system", 
        "content": (
            "You are Javis, an intelligent and friendly voice assistant. Always answer clearly, briefly, and helpfully."
        )
    }

    full_messages = [system_prompt] + trimmed_messages

    # ✅ Use ainvoke instead of invoke
    response = await llm.ainvoke(full_messages)
    
    return {"messages": messages + [response]}
