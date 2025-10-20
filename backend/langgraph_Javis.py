from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

class VoiceState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    # topic: str
    # joke: str
    # explanation: str
    # sr: float
    # bpb: float
    # boundary_percent: float
    # messages: Annotated[list[BaseMessage], add_messages]
    # score: int = Field(description='Score out of 10', ge=0, le=10)
    # topic: str
    # tweet: str
    # evaluation: Literal["approved", "needs_improvement"]
    # tweet_history: Annotated[list[str], operator.add]
    # feedback_history: Annotated[list[str], operator.add]

llm = ChatOpenAI()

def llm_service(state: VoiceState):

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

def tts_service(state: VoiceState):
    return

# Memory
checkpointer = MemorySaver()

# Graph Foundation
graph = StateGraph(VoiceState)

graph.add_node('llm_service', llm_service)
# graph.add_node('tts_service', tts_service)

graph.add_edge(START, 'llm_service')
# graph.add_edge('llm_service', 'ElevenLabs')
graph.add_edge('llm_service', END)

workflow = graph.compile()
chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'

while True:

    user_message = input('User: ')

    if user_message.strip().lower() in ['exit', 'bye', 'quit']:
        break

    config = {'configurable': {'thread_id': thread_id}}

    response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
    print('AI: ', response['messages'][-1].content)