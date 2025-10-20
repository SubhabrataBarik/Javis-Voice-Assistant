from langchain_openai import ChatOpenAI
from VoiceState import VoiceState

# Create the LLM instance
llm = ChatOpenAI()

def llm_service(state: VoiceState):
    """
    LLM service node for LangGraph workflow.
    Takes state.messages, prepends system prompt, and returns model response.
    """

    # take user query from state
    messages = state['messages']
    # Add system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are Javis, an intelligent and friendly voice assistant. Always answer clearly, briefly, and helpfully."
        )
    }

    # Combine system + user messages
    full_messages = [system_prompt] + messages
    # send to llm
    response = llm.invoke(full_messages)
    # response store state
    return {'messages': [response]}