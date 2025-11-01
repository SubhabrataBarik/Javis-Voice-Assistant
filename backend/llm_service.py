from langchain_openai import ChatOpenAI
from VoiceState import VoiceState

# Create the LLM instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=500,
)
TURN_LIMIT = 4  # Number of conversation turns to remember

def llm_service(state: VoiceState):
    """
    LLM service node for LangGraph workflow.
    Takes state.messages, prepends system prompt, and returns model response.
    """

    # take user query from state
    messages = state["messages"]
    # Limit to last TURN_LIMIT * 2 messages (each turn = user + assistant)
    trimmed_messages = messages[-TURN_LIMIT * 2:]

    # Add system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are Javis, an intelligent and friendly voice assistant. Always answer clearly, briefly, and helpfully."
        )
    }

    # Combine system + user messages
    # 4 turns = 8 messages (user+assistant) (4 Turn is having short term memory)
    full_messages = [system_prompt] + trimmed_messages
    # full_messages = [system_prompt] + messages

    # send to llm
    response = llm.invoke(full_messages)
    # response store state
    # return {'messages': [response]}
    return {"messages": messages + [response]}
