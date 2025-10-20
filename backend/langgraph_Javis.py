# -------------------
# Library
# -------------------
from dotenv import load_dotenv
load_dotenv()

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Import the LLM service from separate file
from llm_service import llm_service
from VoiceState import VoiceState

# -------------------
# Define Function
# -------------------
def tts_service(state: VoiceState):
    return

# -------------------
# Setup database/checkpointer
# -------------------
conn = sqlite3.connect(database='Javis.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# -------------------
# Graph definition
# -------------------
graph = StateGraph(VoiceState)

graph.add_node('llm_service', llm_service)
# graph.add_node('tts_service', tts_service)

graph.add_edge(START, 'llm_service')
# graph.add_edge('llm_service', 'ElevenLabs')
graph.add_edge('llm_service', END)

workflow = graph.compile()
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# Chat loop
# -------------------
thread_id = '1'
while True:

    user_message = input('User: ')
    if user_message.strip().lower() in ['exit', 'bye', 'quit']:
        break

    config = {'configurable': {'thread_id': thread_id}}
    response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
    print('AI: ', response['messages'][-1].content)