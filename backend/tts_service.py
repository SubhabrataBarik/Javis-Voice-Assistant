# from VoiceState import VoiceState
# from elevenlabs.client import ElevenLabs
# from elevenlabs.play import play
# import os

# # -------------------
# # Setup ElevenLabs client
# # -------------------
# ffmpeg_path = r"D:\PENDRIVE 32 GB\Game\AI_PROJECTS\javis\ffmpeg-8.0-essentials_build\bin"
# os.environ["PATH"] += os.pathsep + ffmpeg_path

# api_key = os.getenv("ELEVENLABS_API_KEY")
# client = ElevenLabs(api_key=api_key)


# def tts_service(state: VoiceState):
#     """
#     TTS service node for LangGraph workflow.
#     Converts the latest LLM message to speech and plays it.
#     """
#     # Get latest LLM message
#     messages = state['messages'][-1].content

#     # Convert text to audio bytes
#     audio = client.text_to_speech.convert(
#         text=messages,
#         voice_id="JBFqnCBsd6RMkjVDRZzb",
#         model_id="eleven_multilingual_v2",
#         output_format="mp3_44100_128",
#     )

#     # Save audio temporarily (current approach)
#     with open("audio.wav", "wb") as f:
#         for chunk in audio:
#             f.write(chunk)

#     # Play audio
#     with open("audio.wav", "rb") as f:
#         play(f.read())

#     return









import asyncio
import os
import logging
from dotenv import load_dotenv  # ✅ Add this import
from VoiceState import VoiceState
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# ✅ Load environment variables
load_dotenv()

# ✅ Add logging
logger = logging.getLogger(__name__)

# -------------------
# Setup ElevenLabs client
# -------------------
ffmpeg_path = r"D:\PENDRIVE 32 GB\Game\AI_PROJECTS\javis\ffmpeg-8.0-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

api_key = os.getenv("ELEVENLABS_API_KEY")

# ✅ Validate API key exists
if not api_key:
    raise ValueError("❌ Missing ElevenLabs API key. Check your .env file.")

client = ElevenLabs(api_key=api_key)

async def tts_service(state: VoiceState) -> dict:
    """
    Async TTS service with explicit error handling
    """
    audio_file = "audio.wav"  # ✅ Define file path early for cleanup
    
    try:
        # Check if we have messages to speak
        messages = state.get('messages', [])
        if not messages:
            return {
                "status": "error",
                "error": {
                    "type": "no_content",
                    "message": "No message to speak",
                    "recoverable": False
                }
            }

        # Get latest LLM message
        latest_message = messages[-1].content
        
        # ✅ Validate message content
        if not latest_message or not latest_message.strip():
            return {
                "status": "error", 
                "error": {
                    "type": "empty_content",
                    "message": "Message content is empty",
                    "recoverable": False
                }
            }

        logger.info(f"🔊 Converting to speech: {latest_message[:50]}...")

        # ✅ Run TTS with timeout to avoid hanging
        audio = await asyncio.wait_for(
            asyncio.to_thread(
                client.text_to_speech.convert,
                text=latest_message,
                voice_id="JBFqnCBsd6RMkjVDRZzb", 
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            ),
            timeout=30.0  # 30 second timeout
        )

        # Save audio file
        with open(audio_file, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        # ✅ Play audio with timeout
        with open(audio_file, "rb") as f:
            await asyncio.wait_for(
                asyncio.to_thread(play, f.read()),
                timeout=60.0  # Reasonable timeout for playback
            )

        logger.info("🔊 Audio playback completed")

        # ✅ Success state
        return {
            "status": "success",
            "tts_completed": True,
            "message_spoken": latest_message
        }

    except asyncio.TimeoutError:
        # ✅ Specific timeout handling
        return {
            "status": "error",
            "error": {
                "type": "timeout", 
                "message": "TTS operation timed out",
                "recoverable": True,
                "original_error": "Operation exceeded timeout limit"
            }
        }
        
    except ConnectionError as e:
        return {
            "status": "error",
            "error": {
                "type": "connection",
                "message": "Could not connect to TTS service",
                "recoverable": True,
                "original_error": str(e)
            }
        }
        
    except PermissionError as e:
        return {
            "status": "error",
            "error": {
                "type": "permission", 
                "message": "Audio device access denied",
                "recoverable": False,
                "original_error": str(e)
            }
        }
        
    except FileNotFoundError as e:
        # ✅ Handle missing ffmpeg or audio issues
        return {
            "status": "error",
            "error": {
                "type": "audio_system",
                "message": "Audio system not found. Check ffmpeg installation.",
                "recoverable": False,
                "original_error": str(e)
            }
        }
        
    except Exception as e:
        logger.exception("TTS service error")
        return {
            "status": "error",
            "error": {
                "type": "unknown",
                "message": f"TTS error: {str(e)}",
                "recoverable": False,
                "original_error": str(e)
            }
        }
        
    finally:
        # ✅ Clean up audio file
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception as e:
            logger.warning(f"Could not clean up audio file: {e}")
