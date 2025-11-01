# -------------------
# Library
# -------------------
from dotenv import load_dotenv
load_dotenv()

import asyncio
import sqlite3
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # ✅ Change this import
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Import the services from separate files
from VoiceState import VoiceState
from tts_service import tts_service  # Make sure this is also async
from llm_service import llm_service  # Make sure this is also async
from stt_service import stt_service  # Now async

# -------------------
# Async Chat loop with proper database setup
# -------------------
async def main():
    """
    Main async chat loop with async database connection
    """
    thread_id = "2"
    
    print("🤖 Javis Voice Assistant Started!")
    print("🎙️ Say something to begin... (say 'exit', 'bye', or 'quit' to end)")

    # ✅ Use async context manager for AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string("Javis.db") as checkpointer:
        
        # -------------------
        # Graph definition (same as before but inside async context)
        # -------------------
        graph = StateGraph(VoiceState)

        # Add nodes (all async now)
        graph.add_node('stt_service', stt_service)
        graph.add_node('llm_service', llm_service)
        graph.add_node('tts_service', tts_service)

        # Add edges (same as before)
        graph.add_edge(START, 'stt_service')
        graph.add_edge('stt_service', 'llm_service')
        graph.add_edge('llm_service', 'tts_service')
        graph.add_edge('tts_service', END)

        # Compile the graph with async checkpointer
        chatbot = graph.compile(checkpointer=checkpointer)

        while True:
            try:
                config = {"configurable": {"thread_id": thread_id}}

                print("\n🔄 Starting voice workflow...")

                # ✅ Use ainvoke with async checkpointer
                state = await chatbot.ainvoke({}, config=config)
                
                # Extract latest user utterance for exit checking
                user_message = None
                if hasattr(state, "messages") and state.messages:
                    # Get the last human message (skip assistant responses)
                    for msg in reversed(state.messages):
                        if hasattr(msg, 'type') and msg.type == 'human':
                            user_message = msg.content.strip().lower()
                            print(f"🗣️ You said: {msg.content}")
                            break

                # Check for exit conditions
                if user_message and any(exit_word in user_message for exit_word in ["exit", "bye", "quit", "goodbye"]):
                    print("👋 Ending session. Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in voice workflow: {e}")
                print("🔄 Continuing... Try speaking again.")
                continue

async def run_assistant():
    """
    Wrapper function to run the assistant with proper error handling
    """
    try:
        await main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("🔚 Voice Assistant Shutdown Complete")

# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    # ✅ Run the async main function
    try:
        asyncio.run(run_assistant())
    except KeyboardInterrupt:
        print("\n👋 Assistant stopped by user. Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
