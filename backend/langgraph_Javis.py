# # -------------------
# # Library
# # -------------------
# from dotenv import load_dotenv
# load_dotenv()

# import asyncio
# import sqlite3
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import HumanMessage

# # Import the services from separate files
# from VoiceState import VoiceState
# from tts_service import tts_service
# from llm_service import llm_service
# from stt_service import stt_service

# # -------------------
# # Async Chat loop with proper database setup
# # -------------------
# async def main():
#     """
#     Main async chat loop with async database connection
#     """
#     thread_id = "2"
    
#     print("🤖 Javis Voice Assistant Started!")
#     print("🎙️ Say something to begin... (say 'exit', 'bye', or 'quit' to end)")

#     # ✅ Use async context manager for AsyncSqliteSaver
#     async with AsyncSqliteSaver.from_conn_string("Javis.db") as checkpointer:
        
#         # -------------------
#         # Graph definition (same as before but inside async context)
#         # -------------------
#         graph = StateGraph(VoiceState)

#         # Add nodes (all async now)
#         graph.add_node('stt_service', stt_service)
#         graph.add_node('llm_service', llm_service)
#         graph.add_node('tts_service', tts_service)

#         # Add edges (same as before)
#         graph.add_edge(START, 'stt_service')
#         graph.add_edge('stt_service', 'llm_service')
#         graph.add_edge('llm_service', 'tts_service')
#         graph.add_edge('tts_service', END)

#         # Compile the graph with async checkpointer
#         chatbot = graph.compile(checkpointer=checkpointer)

#         while True:
#             try:
#                 config = {"configurable": {"thread_id": thread_id}}

#                 print("\n🔄 Starting voice workflow...")

#                 # ✅ Use ainvoke with async checkpointer
#                 state = await chatbot.ainvoke({}, config=config)
                
#                 # Extract latest user utterance for exit checking
#                 user_message = None
#                 if hasattr(state, "messages") and state.messages:
#                     # Get the last human message (skip assistant responses)
#                     for msg in reversed(state.messages):
#                         if hasattr(msg, 'type') and msg.type == 'human':
#                             user_message = msg.content.strip().lower()
#                             print(f"🗣️ You said: {msg.content}")
#                             break

#                 # Check for exit conditions
#                 if user_message and any(exit_word in user_message for exit_word in ["exit", "bye", "quit", "goodbye"]):
#                     print("👋 Ending session. Goodbye!")
#                     break
                    
#             except KeyboardInterrupt:
#                 print("\n👋 Session interrupted. Goodbye!")
#                 break
#             except Exception as e:
#                 print(f"❌ Error in voice workflow: {e}")
#                 print("🔄 Continuing... Try speaking again.")
#                 continue

# async def run_assistant():
#     """
#     Wrapper function to run the assistant with proper error handling
#     """
#     try:
#         await main()
#     except Exception as e:
#         print(f"❌ Fatal error: {e}")
#     finally:
#         print("🔚 Voice Assistant Shutdown Complete")

# # -------------------
# # Entry Point
# # -------------------
# if __name__ == "__main__":
#     # ✅ Run the async main function
#     try:
#         asyncio.run(run_assistant())
#     except KeyboardInterrupt:
#         print("\n👋 Assistant stopped by user. Goodbye!")
#     except Exception as e:
#         print(f"❌ Failed to start assistant: {e}")











# -------------------
# Library
# -------------------
from dotenv import load_dotenv
load_dotenv()

import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# Import services
from VoiceState import VoiceState
from tts_service import tts_service
from llm_service import llm_service
from stt_service import stt_service

# -------------------
# Simple Error Handler (Optional)
# -------------------
async def handle_service_error(state: VoiceState, service_name: str, error_info: dict) -> dict:
    """Simple error handler that logs and continues"""
    error_message = error_info.get("message", "Unknown error")
    print(f"⚠️ {service_name} Error: {error_message}")
    
    # Add error message to conversation for user awareness
    current_messages = state.get("messages", [])
    error_response = AIMessage(content=f"I encountered an issue with {service_name.lower()}. Let me try to continue...")
    
    return {
        "messages": current_messages + [error_response],
        "status": "error_handled"
    }

# -------------------
# Enhanced Service Wrappers (Keep Simple Flow)
# -------------------
async def enhanced_stt_service(state: VoiceState) -> dict:
    """STT with error handling that doesn't break the flow"""
    result = await stt_service(state)
    
    # If error occurred, handle it but continue
    if result.get("status") == "error":
        error_info = result.get("error", {})
        if error_info.get("recoverable", False):
            print(f"🔄 STT Issue: {error_info.get('message', 'Unknown error')}")
            # Return empty result to continue flow
            return {"messages": state.get("messages", [])}
        else:
            # For non-recoverable errors, add error message but continue
            return await handle_service_error(state, "Speech Recognition", error_info)
    
    return result

async def enhanced_llm_service(state: VoiceState) -> dict:
    """LLM with error handling that doesn't break the flow"""
    result = await llm_service(state)
    
    # If error occurred, provide fallback response
    if result.get("status") == "error":
        error_info = result.get("error", {})
        print(f"🔄 LLM Issue: {error_info.get('message', 'Unknown error')}")
        
        # Provide fallback response
        current_messages = state.get("messages", [])
        fallback_response = AIMessage(content="I'm having trouble processing your request. Could you please try again?")
        
        return {
            "messages": current_messages + [fallback_response],
            "status": "success"  # Continue with fallback
        }
    
    return result

async def enhanced_tts_service(state: VoiceState) -> dict:
    """TTS with error handling that doesn't break the flow"""
    result = await tts_service(state)
    
    # If error occurred, log it but consider workflow complete
    if result.get("status") == "error":
        error_info = result.get("error", {})
        print(f"🔄 TTS Issue: {error_info.get('message', 'Unknown error')}")
        # Still return success to complete workflow
        return {"status": "success", "tts_completed": False}
    
    return result

# -------------------
# Main Application
# -------------------
async def main():
    """
    Simple main loop with enhanced error handling
    """
    thread_id = "2"
    conversation_count = 0
    
    print("=" * 60)
    print("🤖 JAVIS VOICE ASSISTANT")
    print("=" * 60)
    print("🚀 Initializing Javis Voice Assistant...")
    print("🤖 Javis Voice Assistant Started!")
    print("🎙️ Say something to begin conversation...")
    print("💡 Say 'exit', 'bye', 'quit', or 'goodbye' to end")
    print("=" * 50)

    async with AsyncSqliteSaver.from_conn_string("Javis.db") as checkpointer:
        
        # ✅ Simple Linear Graph (No Complex Routing)
        graph = StateGraph(VoiceState)

        # Add service nodes (enhanced versions with error handling)
        graph.add_node('stt_service', enhanced_stt_service)
        graph.add_node('llm_service', enhanced_llm_service)  
        graph.add_node('tts_service', enhanced_tts_service)

        # ✅ Simple Linear Flow (Just Like Before)
        graph.add_edge(START, 'stt_service')
        graph.add_edge('stt_service', 'llm_service')
        graph.add_edge('llm_service', 'tts_service') 
        graph.add_edge('tts_service', END)

        # Compile graph
        chatbot = graph.compile(checkpointer=checkpointer)

        # Main conversation loop
        while True:
            try:
                conversation_count += 1
                config = {"configurable": {"thread_id": thread_id}}
                
                print(f"\n🔄 Starting conversation #{conversation_count}...")

                # ✅ Simple linear execution (STT -> LLM -> TTS)
                state = await chatbot.ainvoke({}, config=config)
                
                # Extract and display conversation
                messages = state.get("messages", [])
                if messages:
                    # Show conversation summary
                    print("\n📝 Conversation Summary:")
                    human_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type == 'human']
                    ai_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type == 'ai']
                    
                    # Display recent messages
                    recent_messages = messages[-4:]  # Last 4 messages
                    for msg in recent_messages:
                        if hasattr(msg, 'type'):
                            if msg.type == 'human':
                                print(f"   🗣️ You: {msg.content}")
                            elif msg.type == 'ai':
                                print(f"   🤖 Javis: {msg.content}")
                    
                    # Check for exit commands
                    should_exit = False
                    for msg in reversed(messages):
                        if hasattr(msg, 'type') and msg.type == 'human':
                            content = msg.content.strip().lower()
                            if any(exit_word in content for exit_word in ["exit", "bye", "quit", "goodbye"]):
                                print("\n👋 Ending session. Goodbye!")
                                should_exit = True
                            break
                    
                    if should_exit:
                        break
                
                # Brief pause between conversations
                print("\n⏳ Ready for next conversation in 2 seconds...")
                await asyncio.sleep(2)
                    
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in conversation #{conversation_count}: {e}")
                print("🔄 Continuing to next conversation...")
                await asyncio.sleep(1)
                continue

async def run_assistant():
    """Enhanced wrapper with better error handling"""
    try:
        await main()
    except Exception as e:
        print(f"❌ Fatal application error: {e}")
    finally:
        print("🔚 Voice Assistant Shutdown Complete")
        print("Thank you for using Javis! 🤖✨")

if __name__ == "__main__":
    try:
        asyncio.run(run_assistant())
    except KeyboardInterrupt:
        print("\n👋 Assistant stopped by user. Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
