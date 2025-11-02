import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from VoiceState import VoiceState

load_dotenv()
logger = logging.getLogger(__name__)

# Create the LLM instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=500,
)
TURN_LIMIT = 4

async def llm_service(state: VoiceState) -> dict:
    """
    Async LLM service with error handling that doesn't break flow
    """
    try:
        messages = state.get("messages", [])
        if not messages:
            return {
                "status": "error",
                "error": {
                    "type": "no_input",
                    "message": "No messages to process",
                    "recoverable": False
                }
            }

        trimmed_messages = messages[-TURN_LIMIT * 2:]

        system_prompt = {
            "role": "system", 
            "content": (
                "You are Javis, an intelligent and friendly voice assistant. Always answer clearly, briefly, and helpfully."
            )
        }

        full_messages = [system_prompt] + trimmed_messages

        # ✅ Use ainvoke with reasonable timeout
        response = await asyncio.wait_for(
            llm.ainvoke(full_messages), 
            timeout=30.0
        )
        
        # ✅ Success state
        return {
            "messages": messages + [response],
            "status": "success"
        }
        
    except asyncio.TimeoutError:
        logger.error("LLM timeout")
        return {
            "status": "error",
            "error": {
                "type": "timeout",
                "message": "LLM response took too long",
                "recoverable": True
            }
        }
        
    except Exception as e:
        logger.exception(f"LLM error: {e}")
        return {
            "status": "error",
            "error": {
                "type": "unknown",
                "message": f"LLM error: {str(e)}",
                "recoverable": True,
                "original_error": str(e)
            }
        }
