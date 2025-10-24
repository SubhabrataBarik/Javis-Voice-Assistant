# # -------------------
# # Library
# # -------------------
# from dotenv import load_dotenv
# load_dotenv()

# import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import HumanMessage

# # Import the LLM service from separate file
# from VoiceState import VoiceState
# from tts_service import tts_service
# from llm_service import llm_service
# from stt_service import stt_service


# # -------------------
# # Define Function
# # -------------------
# def stt_service(state: VoiceState):
#     return


# # -------------------
# # Setup database/checkpointer
# # -------------------
# conn = sqlite3.connect(database='Javis.db', check_same_thread=False)
# # Checkpointer
# checkpointer = SqliteSaver(conn=conn)


# # -------------------
# # Graph definition
# # -------------------
# graph = StateGraph(VoiceState)

# graph.add_node('stt_service', stt_service)
# graph.add_node('llm_service', llm_service)
# graph.add_node('tts_service', tts_service)

# graph.add_edge(START, 'stt_service')
# graph.add_edge('stt_service', 'llm_service')
# graph.add_edge('llm_service', 'tts_service')
# graph.add_edge('tts_service', END)

# workflow = graph.compile()
# chatbot = graph.compile(checkpointer=checkpointer)


# # -------------------
# # Chat loop
# # -------------------
# thread_id = '1'
# while True:

#     user_message = input('User: ')
#     if user_message.strip().lower() in ['exit', 'bye', 'quit']:
#         break

#     config = {'configurable': {'thread_id': thread_id}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print('AI: ', response['messages'][-1].content)




















# langgraph.py
from dotenv import load_dotenv
load_dotenv()

import sqlite3
import logging
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from VoiceState import VoiceState
from tts_service import tts_service
from llm_service import llm_service
from stt_service import stt_service  # make sure this is your real STT implementation

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup DB/checkpointer
conn = sqlite3.connect(database='Javis.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build graph
graph = StateGraph(VoiceState)
graph.add_node('stt_service', stt_service)
graph.add_node('llm_service', llm_service)
graph.add_node('tts_service', tts_service)
graph.add_edge(START, 'stt_service')
graph.add_edge('stt_service', 'llm_service')
graph.add_edge('llm_service', 'tts_service')
graph.add_edge('tts_service', END)

# Compile once
chatbot = graph.compile(checkpointer=checkpointer)

def _print_assistant_reply(result):
    try:
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            print("AI:", result["messages"][-1].content)
            return
        messages = getattr(result, "messages", None)
        if messages and isinstance(messages, (list, tuple)) and messages:
            last = messages[-1]
            content = getattr(last, "content", None) or (last if isinstance(last, str) else None)
            if content:
                print("AI:", content)
                return
    except Exception as e:
        logger.debug("Error extracting assistant reply: %s", e)
    print("AI (raw result):", result)

def run_text_mode(thread_id="1"):
    print("Text mode — type your message and press Enter. Type 'exit' to go back to mode selection.")
    while True:
        txt = input("You (text): ").strip()
        if txt.lower() in ["exit", "quit", "bye"]:
            print("Returning to mode selection.")
            break
        config = {"configurable": {"thread_id": thread_id}}
        try:
            payload = {"messages": [HumanMessage(content=txt)]}
            res = chatbot.invoke(payload, config=config)
            _print_assistant_reply(res)
        except Exception as e:
            logger.exception("Error invoking graph in text mode: %s", e)
            print("Error:", e)

def run_play_mode(thread_id="1"):
    print("Play mode — will trigger stt_service -> llm_service -> tts_service. Type 'exit' to go back to mode selection.")
    while True:
        cmd = input("Press Enter to record (or type 'exit' to stop play mode): ").strip()
        if cmd.lower() in ["exit", "quit", "bye"]:
            print("Returning to mode selection.")
            break

        config = {"configurable": {"thread_id": thread_id}}
        # Try multiple invoke styles for compatibility
        tried = []
        result = None
        try:
            tried.append("invoke({})")
            result = chatbot.invoke({}, config=config)
        except Exception as e1:
            logger.debug("invoke({}) failed: %s", e1)
            try:
                tried.append("invoke(None)")
                result = chatbot.invoke(None, config=config)
            except Exception as e2:
                logger.debug("invoke(None) failed: %s", e2)
                try:
                    tried.append("invoke({'start': True})")
                    result = chatbot.invoke({'start': True}, config=config)
                except Exception as e3:
                    logger.exception("All invoke attempts failed: %s / %s / %s", e1, e2, e3)
                    print("Error: could not trigger voice flow. See logs.")
                    continue

        _print_assistant_reply(result)

def main():
    print("Choose mode: (text) typed input or (play) record audio via STT.")
    while True:
        choice = input("Select mode [text/play] (or 'exit' to quit): ").strip().lower()
        if choice in ["exit", "quit", "bye"]:
            print("Exiting.")
            break
        if choice not in ["text", "play", ""]:
            print("Invalid choice; type 'text' or 'play'.")
            continue
        # default empty -> text
        if choice == "" or choice == "text":
            run_text_mode()
        else:
            run_play_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print("Clean exit.")
