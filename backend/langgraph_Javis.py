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
# Smart Delay Configuration
# -------------------
class SmartDelayConfig:
    """Configuration for smart dynamic delays"""
    # Base delay settings (in seconds)
    SHORT_RESPONSE_DELAY = 0.3   # For responses < 50 characters
    MEDIUM_RESPONSE_DELAY = 0.5  # For responses 50-150 characters  
    LONG_RESPONSE_DELAY = 1.5    # For responses > 150 characters
    
    # Response length thresholds (in characters)
    SHORT_THRESHOLD = 50
    MEDIUM_THRESHOLD = 150
    
    # Minimum and maximum delays (safety bounds)
    MIN_DELAY = 0.2
    MAX_DELAY = 3.0
    
    # Special case delays
    NO_RESPONSE_DELAY = 0.3      # When no AI response
    ERROR_DELAY = 1.0            # When there's an error
    
    @classmethod
    def calculate_delay(cls, response_text: str = "", error_occurred: bool = False) -> float:
        """
        Calculate smart delay based on response characteristics
        """
        if error_occurred:
            return cls.ERROR_DELAY
            
        if not response_text:
            return cls.NO_RESPONSE_DELAY
            
        response_length = len(response_text.strip())
        
        if response_length < cls.SHORT_THRESHOLD:
            delay = cls.SHORT_RESPONSE_DELAY
        elif response_length < cls.MEDIUM_THRESHOLD:
            delay = cls.MEDIUM_RESPONSE_DELAY
        else:
            delay = cls.LONG_RESPONSE_DELAY
            
        # Apply safety bounds
        delay = max(cls.MIN_DELAY, min(delay, cls.MAX_DELAY))
        
        return delay

# -------------------
# Enhanced Error Handler with Smart Delay
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
# Enhanced Service Wrappers with Smart Delay Support
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
            return {"messages": state.get("messages", []), "stt_error": True}
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
            "status": "success",  # Continue with fallback
            "llm_error": True
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
        return {"status": "success", "tts_completed": False, "tts_error": True}
    
    return result

# -------------------
# Smart Conversation Analysis
# -------------------
class ConversationAnalyzer:
    """Analyzes conversation for smart delay calculation"""
    
    @staticmethod
    def get_last_ai_response(messages: list) -> str:
        """Extract the last AI response from messages"""
        if not messages:
            return ""
            
        # Find the last AI message
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                return getattr(msg, 'content', '')
        return ""
    
    @staticmethod
    def analyze_conversation_state(state: VoiceState) -> dict:
        """Analyze conversation state for smart decisions"""
        messages = state.get("messages", [])
        
        # Get last responses
        last_ai_response = ConversationAnalyzer.get_last_ai_response(messages)
        
        # Check for errors in this conversation turn
        has_errors = (
            state.get("stt_error", False) or
            state.get("llm_error", False) or  
            state.get("tts_error", False)
        )
        
        # Analyze response characteristics
        response_analysis = {
            "last_ai_response": last_ai_response,
            "response_length": len(last_ai_response) if last_ai_response else 0,
            "has_errors": has_errors,
            "is_question": "?" in last_ai_response,
            "is_long_explanation": len(last_ai_response) > 200,
            "is_simple_answer": len(last_ai_response) < 30,
        }
        
        return response_analysis

# -------------------
# Main Application with Smart Dynamic Delay
# -------------------
async def main():
    """
    Enhanced main loop with Smart Dynamic Delay system
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
    print("🧠 Smart Dynamic Delay: Adapts to response length")
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

        # Main conversation loop with Smart Dynamic Delay
        while True:
            try:
                conversation_count += 1
                config = {"configurable": {"thread_id": thread_id}}
                
                print(f"\n🔄 Starting conversation #{conversation_count}...")

                # ✅ Simple linear execution (STT -> LLM -> TTS)
                state = await chatbot.ainvoke({}, config=config)
                
                # ✅ Smart Dynamic Delay Analysis
                conversation_analysis = ConversationAnalyzer.analyze_conversation_state(state)
                
                # Extract and display conversation
                messages = state.get("messages", [])
                if messages:
                    # Show conversation summary
                    print("\n📝 Conversation Summary:")
                    
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
                
                # ✅ SMART DYNAMIC DELAY - The magic happens here!
                last_ai_response = conversation_analysis["last_ai_response"]
                has_errors = conversation_analysis["has_errors"]
                
                # Calculate smart delay
                smart_delay = SmartDelayConfig.calculate_delay(
                    response_text=last_ai_response,
                    error_occurred=has_errors
                )
                
                # Enhanced user feedback about the delay
                delay_reason = ""
                if has_errors:
                    delay_reason = " (error recovery)"
                elif conversation_analysis["is_simple_answer"]:
                    delay_reason = " (quick response)"
                elif conversation_analysis["is_long_explanation"]:
                    delay_reason = " (processing time for long response)"
                elif conversation_analysis["is_question"]:
                    delay_reason = " (thinking time)"
                else:
                    delay_reason = " (standard pause)"
                
                print(f"\n⏳ Ready for next conversation in {smart_delay:.1f} seconds{delay_reason}...")
                print(f"🧠 Response analysis: {conversation_analysis['response_length']} chars, "
                      f"{'with errors' if has_errors else 'no errors'}")
                
                # Apply the smart delay
                await asyncio.sleep(smart_delay)
                    
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in conversation #{conversation_count}: {e}")
                print("🔄 Continuing to next conversation...")
                # Use error delay for recovery
                await asyncio.sleep(SmartDelayConfig.ERROR_DELAY)
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
