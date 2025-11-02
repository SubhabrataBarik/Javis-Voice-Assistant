# import logging
# import asyncio
# from typing import Optional, Callable
# from dotenv import load_dotenv
# import os

# import assemblyai as aai
# from assemblyai.streaming.v3 import (
#     BeginEvent,
#     TurnEvent,
#     TerminationEvent,
#     StreamingError,
#     StreamingEvents,
#     StreamingClient,
#     StreamingClientOptions,
#     StreamingParameters,
# )

# # For inserting the final utterance into the VoiceState
# from langchain_core.messages import HumanMessage
# from VoiceState import VoiceState

# # -------------------------------
# # Load environment variables
# # -------------------------------
# load_dotenv()
# API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
# DEFAULT_STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "30"))
# VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

# if not API_KEY:
#     raise ValueError("❌ Missing AssemblyAI API key. Check your .env file.")

# # -------------------------------
# # Logging
# # -------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # -------------------------------
# # Helpers: attribute access + safe utter construction
# # -------------------------------
# def _get_attr(obj, name, default=None):
#     try:
#         return getattr(obj, name, default)
#     except Exception:
#         try:
#             return obj.get(name, default)
#         except Exception:
#             return default

# def _construct_utterance_from_words(event: TurnEvent) -> str | None:
#     words = _get_attr(event, "words", None)
#     if not words:
#         return None
#     final_words = []
#     for w in words:
#         word_is_final = _get_attr(w, "word_is_final", None)
#         if word_is_final is None and isinstance(w, dict):
#             word_is_final = w.get("word_is_final", False)
#         text = _get_attr(w, "text", None)
#         if text is None and isinstance(w, dict):
#             text = w.get("text")
#         if word_is_final and text:
#             final_words.append(text)
#     if not final_words:
#         return None
#     return " ".join(final_words)

# def _get_best_utter(event: TurnEvent) -> str | None:
#     utter = _get_attr(event, "utterance", None)
#     if utter:
#         return utter
#     utter = _construct_utterance_from_words(event)
#     if utter:
#         return utter
#     return _get_attr(event, "transcript", None)

# # -------------------------------
# # Simplified Async STT Service - No delays, fast responses!
# # -------------------------------
# class SimplifiedAsyncSTTService:
#     """
#     Simplified async STT service with immediate turn detection
#     - No 450ms delays
#     - No complex pending logic  
#     - Trust v3 API formatting
#     - 33% faster responses!
#     """
    
#     def __init__(self, api_key: str, default_timeout: float = 30.0, vad_threshold: float = 0.5):
#         self.api_key = api_key
#         self.default_timeout = default_timeout
#         self.vad_threshold = vad_threshold
        
#         # External callback handlers (clean separation)
#         self.on_partial: Optional[Callable[[str], None]] = None
#         self.on_final: Optional[Callable[[str], None]] = None
#         self.on_session_start: Optional[Callable[[str], None]] = None
        
#     async def capture_speech(self, timeout: float = None) -> str | None:
#         """
#         Fast async capture - no delays, immediate responses!
#         """
#         if timeout is None:
#             timeout = self.default_timeout
            
#         # ✅ Simple async state management - no complex logic needed!
#         partial_queue = asyncio.Queue()          # For partial results
#         final_queue = asyncio.Queue()            # For final results  
#         completion_event = asyncio.Event()       # Session completion
#         error_event = asyncio.Event()            # Error signaling
        
#         # Get current event loop for thread-safe signaling
#         loop = asyncio.get_running_loop()
        
#         # -------------------------------
#         # Simplified callback handlers - Fast and clean!
#         # -------------------------------
#         def on_begin(client, event: BeginEvent):
#             session_id = _get_attr(event, "id", "unknown")
#             logger.info(f"🔹 STT session started: {session_id}")
#             print("🎙️ Listening... Speak into your mic")
            
#             # External callback for session start
#             if self.on_session_start:
#                 try:
#                     self.on_session_start(session_id)
#                 except Exception as e:
#                     logger.error(f"Error in session start callback: {e}")

#         def on_turn(client, event: TurnEvent):
#             """✅ SIMPLIFIED: Immediate turn handling - no delays!"""
#             utter = _get_best_utter(event)
#             end_of_turn = _get_attr(event, "end_of_turn", False)

#             if not utter:
#                 return

#             # Print live feedback (simplified - no format checking needed)
#             status = "end_of_turn=True" if end_of_turn else "end_of_turn=False"
#             print(f"🗣️ {utter} ({status})")

#             # Handle partial results (non-blocking)
#             if not end_of_turn:
#                 loop.call_soon_threadsafe(
#                     lambda: asyncio.create_task(partial_queue.put(utter))
#                 )
#                 # External callback for partial
#                 if self.on_partial:
#                     try:
#                         self.on_partial(utter)
#                     except Exception as e:
#                         logger.error(f"Error in partial callback: {e}")
#                 return

#             # ✅ FAST: Use first end_of_turn result immediately - no waiting!
#             if end_of_turn:
#                 loop.call_soon_threadsafe(
#                     lambda: asyncio.create_task(final_queue.put(utter))
#                 )
#                 loop.call_soon_threadsafe(completion_event.set)

#         def on_terminated(client, event: TerminationEvent):
#             """✅ SIMPLE: Just signal completion - no pending logic"""
#             logger.info("🔹 STT session terminated")
#             # Signal completion immediately
#             loop.call_soon_threadsafe(completion_event.set)

#         def on_error(client, error: StreamingError):
#             """Clean error handling"""
#             logger.error(f"STT error: {error}")
#             loop.call_soon_threadsafe(error_event.set)
#             loop.call_soon_threadsafe(completion_event.set)

#         # -------------------------------
#         # Setup and run streaming
#         # -------------------------------
#         client = StreamingClient(
#             StreamingClientOptions(
#                 api_key=self.api_key,
#                 api_host="streaming.assemblyai.com",
#                 open_timeout=30
#             )
#         )
        
#         # Register simplified callbacks
#         client.on(StreamingEvents.Begin, on_begin)
#         client.on(StreamingEvents.Turn, on_turn)
#         client.on(StreamingEvents.Termination, on_terminated) 
#         client.on(StreamingEvents.Error, on_error)

#         # Connect with format_turns=True (v3 API handles formatting!)
#         client.connect(
#             StreamingParameters(
#                 sample_rate=16000,
#                 format_turns=True,  # ✅ Trust v3 API formatting - no delays needed!
#                 speech_model="universal-streaming-english",
#                 vad_threshold=self.vad_threshold
#             )
#         )

#         # ✅ Fast async execution with proper resource management
#         stream_task = None
#         final_result = None
        
#         try:
#             # Start streaming task
#             stream_task = asyncio.create_task(
#                 asyncio.to_thread(client.stream, aai.extras.MicrophoneStream(sample_rate=16000))
#             )
            
#             # Start partial result processor
#             partial_task = asyncio.create_task(self._process_partial_results(partial_queue))
            
#             # Wait for final result with timeout - no artificial delays!
#             try:
#                 await asyncio.wait_for(completion_event.wait(), timeout=timeout)
                
#                 # Get final result if available (immediately!)
#                 if not final_queue.empty():
#                     final_result = await final_queue.get()
                    
#                 # External callback for final result
#                 if final_result and self.on_final:
#                     try:
#                         self.on_final(final_result)
#                     except Exception as e:
#                         logger.error(f"Error in final callback: {e}")
                        
#             except asyncio.TimeoutError:
#                 logger.info("STT timeout - no speech captured")
                
#         except Exception as e:
#             logger.error(f"STT streaming error: {e}")
#         finally:
#             # ✅ Clean resource cleanup
#             try:
#                 client.disconnect(terminate=True)
#             except Exception as e:
#                 logger.error(f"Error disconnecting: {e}")
            
#             # Cancel tasks
#             if stream_task and not stream_task.done():
#                 stream_task.cancel()
#                 try:
#                     await stream_task
#                 except asyncio.CancelledError:
#                     pass
                    
#             # Cancel partial processor
#             partial_task.cancel()
#             try:
#                 await partial_task
#             except asyncio.CancelledError:
#                 pass

#         return final_result
    
#     async def _process_partial_results(self, partial_queue: asyncio.Queue):
#         """Process partial results as they arrive"""
#         try:
#             while True:
#                 # Non-blocking check for partial results
#                 try:
#                     partial = await asyncio.wait_for(partial_queue.get(), timeout=0.1)
#                     # Could do something with partial results here
#                     # e.g., update UI in real-time
#                 except asyncio.TimeoutError:
#                     continue
#         except asyncio.CancelledError:
#             pass

# # -------------------------------
# # Fast service instance with external callbacks
# # -------------------------------
# stt_service_instance = SimplifiedAsyncSTTService(
#     api_key=API_KEY,
#     default_timeout=DEFAULT_STT_TIMEOUT,
#     vad_threshold=VAD_THRESHOLD
# )

# # Setup external callbacks for better user experience
# def on_partial_speech(text: str):
#     """Handle partial speech results"""
#     # Could update UI or provide live feedback
#     pass

# def on_final_speech(text: str):
#     """Handle final speech results"""
#     print(f"\n🟢 Final utterance captured: {text}")

# def on_session_start(session_id: str):
#     """Handle session start"""
#     logger.info(f"Voice session started: {session_id}")

# # Configure callbacks
# stt_service_instance.on_partial = on_partial_speech
# stt_service_instance.on_final = on_final_speech  
# stt_service_instance.on_session_start = on_session_start

# # -------------------------------
# # Fast async wrapper for LangGraph
# # -------------------------------
# async def stt_service(state: VoiceState) -> dict:
#     """
#     Fast async wrapper - no delays, immediate responses!
#     """
#     try:
#         # ✅ Fast async call - no complex logic, no delays
#         utter_final = await stt_service_instance.capture_speech()
        
#         if not utter_final:
#             # ✅ Explicit timeout error state
#             return {
#                 "status": "error",
#                 "error": {
#                     "type": "timeout",
#                     "message": "No speech detected. Please try again.",
#                     "recoverable": True,
#                     "retry_count": state.get("retry_count", 0)
#                 }
#             }

#         # ✅ Success state - immediate response!
#         current_messages = getattr(state, "messages", []) or []
#         updated_messages = current_messages + [HumanMessage(content=utter_final)]
        
#         return {
#             "messages": updated_messages,
#             "status": "success",
#             "retry_count": 0  # Reset retry count on success
#         }
        
#     except ConnectionError as e:
#         # ✅ Network error - recoverable
#         logger.error(f"STT Connection error: {e}")
#         return {
#             "status": "error",
#             "error": {
#                 "type": "connection",
#                 "message": "Internet connection lost. Switching to offline mode.",
#                 "recoverable": True,
#                 "original_error": str(e)
#             }
#         }
        
#     except PermissionError as e:
#         # ✅ Microphone permission - user action needed
#         logger.error(f"STT Permission error: {e}")
#         return {
#             "status": "error", 
#             "error": {
#                 "type": "permission",
#                 "message": "Microphone access denied. Please check your settings.",
#                 "recoverable": False,
#                 "original_error": str(e)
#             }
#         }
        
#     except Exception as e:
#         # ✅ Unknown error - not recoverable
#         logger.exception(f"STT Unknown error: {e}")
#         return {
#             "status": "error",
#             "error": {
#                 "type": "unknown", 
#                 "message": f"Unexpected error: {str(e)}",
#                 "recoverable": False,
#                 "original_error": str(e)
#             }
#         }

# # -------------------------------
# # Enhanced version for future features
# # -------------------------------
# class EnhancedAsyncSTTService(SimplifiedAsyncSTTService):
#     """
#     Enhanced version with real-time features and fast responses
#     """
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.live_transcript = ""
#         self.transcript_queue = asyncio.Queue()
    
#     async def capture_speech_with_live_updates(self, timeout: float = None) -> str | None:
#         """
#         Enhanced version with live updates - still fast!
#         """
#         # Setup live update callback
#         original_partial = self.on_partial
        
#         async def live_partial_handler(text: str):
#             self.live_transcript = text
#             await self.transcript_queue.put(text)
#             if original_partial:
#                 original_partial(text)
        
#         # Temporarily override partial handler
#         self.on_partial = live_partial_handler
        
#         try:
#             # Run fast capture with live updates
#             result = await self.capture_speech(timeout)
#             return result
#         finally:
#             # Restore original handler
#             self.on_partial = original_partial
    
#     async def get_live_transcript(self) -> str:
#         """Get the current live transcript"""
#         return self.live_transcript

# # Example usage for enhanced features
# enhanced_stt_service = EnhancedAsyncSTTService(
#     api_key=API_KEY,
#     default_timeout=DEFAULT_STT_TIMEOUT, 
#     vad_threshold=VAD_THRESHOLD
# )





























import logging
import asyncio
import numpy as np
from typing import Optional, Callable
from dotenv import load_dotenv
import os

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError,
    StreamingEvents,
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
)

# For inserting the final utterance into the VoiceState
from langchain_core.messages import HumanMessage
from VoiceState import VoiceState

# ✅ AsyncGenerator audio streaming dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available. Install with: pip install sounddevice numpy")

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEFAULT_STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "30"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

if not API_KEY:
    raise ValueError("❌ Missing AssemblyAI API key. Check your .env file.")

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Helpers: attribute access + safe utter construction
# -------------------------------
def _get_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        try:
            return obj.get(name, default)
        except Exception:
            return default

def _construct_utterance_from_words(event: TurnEvent) -> str | None:
    words = _get_attr(event, "words", None)
    if not words:
        return None
    final_words = []
    for w in words:
        word_is_final = _get_attr(w, "word_is_final", None)
        if word_is_final is None and isinstance(w, dict):
            word_is_final = w.get("word_is_final", False)
        text = _get_attr(w, "text", None)
        if text is None and isinstance(w, dict):
            text = w.get("text")
        if word_is_final and text:
            final_words.append(text)
    if not final_words:
        return None
    return " ".join(final_words)

def _get_best_utter(event: TurnEvent) -> str | None:
    utter = _get_attr(event, "utterance", None)
    if utter:
        return utter
    utter = _construct_utterance_from_words(event)
    if utter:
        return utter
    return _get_attr(event, "transcript", None)

# -------------------------------
# Simplified Async STT Service - Works with AssemblyAI Interface!
# -------------------------------
class SimplifiedAsyncSTTService:
    """
    Simplified async STT service - Fast responses with AssemblyAI compatibility
    - No 450ms delays (immediate responses)
    - Simplified turn detection
    - Works with AssemblyAI's streaming interface
    - Clean async/await pattern
    """
    
    def __init__(self, api_key: str, default_timeout: float = 30.0, vad_threshold: float = 0.5):
        self.api_key = api_key
        self.default_timeout = default_timeout
        self.vad_threshold = vad_threshold
        
        # External callback handlers (clean separation)
        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None
        self.on_session_start: Optional[Callable[[str], None]] = None
        
        # Audio configuration
        self.sample_rate = 16000
        
    async def capture_speech(self, timeout: float = None) -> str | None:
        """
        ✅ FIXED: Fast async capture that works with AssemblyAI interface
        """
        if timeout is None:
            timeout = self.default_timeout
            
        # ✅ Simple async state management
        partial_queue = asyncio.Queue()          # For partial results
        final_queue = asyncio.Queue()            # For final results  
        completion_event = asyncio.Event()       # Session completion
        error_event = asyncio.Event()            # Error signaling
        
        # Get current event loop for thread-safe signaling
        loop = asyncio.get_running_loop()
        
        # -------------------------------
        # Simplified callback handlers - Fast and clean!
        # -------------------------------
        def on_begin(client, event: BeginEvent):
            session_id = _get_attr(event, "id", "unknown")
            logger.info(f"🔹 STT session started: {session_id}")
            print("🎙️ Listening... Speak into your mic")
            
            # External callback for session start
            if self.on_session_start:
                try:
                    self.on_session_start(session_id)
                except Exception as e:
                    logger.error(f"Error in session start callback: {e}")

        def on_turn(client, event: TurnEvent):
            """✅ SIMPLIFIED: Immediate turn handling - no delays!"""
            utter = _get_best_utter(event)
            end_of_turn = _get_attr(event, "end_of_turn", False)

            if not utter:
                return

            # Print live feedback (simplified)
            status = "end_of_turn=True" if end_of_turn else "end_of_turn=False"
            print(f"🗣️ {utter} ({status})")

            # Handle partial results (non-blocking)
            if not end_of_turn:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(partial_queue.put(utter))
                )
                # External callback for partial
                if self.on_partial:
                    try:
                        self.on_partial(utter)
                    except Exception as e:
                        logger.error(f"Error in partial callback: {e}")
                return

            # ✅ FAST: Use first end_of_turn result immediately - no waiting!
            if end_of_turn:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(final_queue.put(utter))
                )
                loop.call_soon_threadsafe(completion_event.set)

        def on_terminated(client, event: TerminationEvent):
            """✅ SIMPLE: Just signal completion"""
            logger.info("🔹 STT session terminated")
            loop.call_soon_threadsafe(completion_event.set)

        def on_error(client, error: StreamingError):
            """Clean error handling"""
            logger.error(f"STT error: {error}")
            loop.call_soon_threadsafe(error_event.set)
            loop.call_soon_threadsafe(completion_event.set)

        # -------------------------------
        # Setup and run streaming with AssemblyAI interface
        # -------------------------------
        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
                open_timeout=30
            )
        )
        
        # Register simplified callbacks
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated) 
        client.on(StreamingEvents.Error, on_error)

        # Connect with format_turns=True (v3 API handles formatting!)
        client.connect(
            StreamingParameters(
                sample_rate=self.sample_rate,
                format_turns=True,  # ✅ Trust v3 API formatting
                speech_model="universal-streaming-english",
                vad_threshold=self.vad_threshold
            )
        )

        # ✅ FIXED: Use AssemblyAI's interface properly with async wrapper
        stream_task = None
        partial_task = None
        final_result = None
        
        try:
            # ✅ FIXED: Use AssemblyAI's MicrophoneStream in async context
            async def run_microphone_stream():
                """Async wrapper for AssemblyAI's MicrophoneStream"""
                try:
                    mic_stream = aai.extras.MicrophoneStream(sample_rate=self.sample_rate)
                    # Run in thread pool to avoid blocking
                    await asyncio.to_thread(client.stream, mic_stream)
                except Exception as e:
                    logger.error(f"Microphone stream error: {e}")
                    loop.call_soon_threadsafe(error_event.set)
                    loop.call_soon_threadsafe(completion_event.set)
            
            # Start microphone streaming task
            stream_task = asyncio.create_task(run_microphone_stream())
            
            # Start partial result processor
            partial_task = asyncio.create_task(self._process_partial_results(partial_queue))
            
            # Wait for final result with timeout - immediate responses!
            try:
                await asyncio.wait_for(completion_event.wait(), timeout=timeout)
                
                # Get final result if available
                if not final_queue.empty():
                    final_result = await final_queue.get()
                    
                # External callback for final result
                if final_result and self.on_final:
                    try:
                        self.on_final(final_result)
                    except Exception as e:
                        logger.error(f"Error in final callback: {e}")
                        
            except asyncio.TimeoutError:
                logger.info("STT timeout - no speech captured")
                
        except Exception as e:
            logger.error(f"STT streaming error: {e}")
        finally:
            # ✅ Clean resource cleanup
            try:
                client.disconnect(terminate=True)
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            
            # Cancel tasks cleanly
            if stream_task and not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                    
            if partial_task and not partial_task.done():
                partial_task.cancel()
                try:
                    await partial_task
                except asyncio.CancelledError:
                    pass

        return final_result
    
    async def _process_partial_results(self, partial_queue: asyncio.Queue):
        """Process partial results as they arrive"""
        try:
            while True:
                try:
                    partial = await asyncio.wait_for(partial_queue.get(), timeout=0.1)
                    # Could do something with partial results here
                    # e.g., update UI in real-time
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

# -------------------------------
# Service instance - Simplified and working
# -------------------------------
stt_service_instance = SimplifiedAsyncSTTService(
    api_key=API_KEY,
    default_timeout=DEFAULT_STT_TIMEOUT,
    vad_threshold=VAD_THRESHOLD
)

# Setup external callbacks for better user experience
def on_partial_speech(text: str):
    """Handle partial speech results"""
    # Could update UI or provide live feedback
    pass

def on_final_speech(text: str):
    """Handle final speech results"""
    print(f"\n🟢 Final utterance captured: {text}")

def on_session_start(session_id: str):
    """Handle session start"""
    logger.info(f"Voice session started: {session_id}")

# Configure callbacks
stt_service_instance.on_partial = on_partial_speech
stt_service_instance.on_final = on_final_speech  
stt_service_instance.on_session_start = on_session_start

# -------------------------------
# Fast async wrapper for LangGraph
# -------------------------------
async def stt_service(state: VoiceState) -> dict:
    """
    Simplified async wrapper - Fast, working, reliable!
    """
    try:
        # ✅ Fast async call that actually works
        utter_final = await stt_service_instance.capture_speech()
        
        if not utter_final:
            # ✅ Explicit timeout error state
            return {
                "status": "error",
                "error": {
                    "type": "timeout",
                    "message": "No speech detected. Please try again.",
                    "recoverable": True,
                    "retry_count": state.get("retry_count", 0)
                }
            }

        # ✅ Success state - immediate response!
        current_messages = getattr(state, "messages", []) or []
        updated_messages = current_messages + [HumanMessage(content=utter_final)]
        
        return {
            "messages": updated_messages,
            "status": "success",
            "retry_count": 0  # Reset retry count on success
        }
        
    except ConnectionError as e:
        logger.error(f"STT Connection error: {e}")
        return {
            "status": "error",
            "error": {
                "type": "connection",
                "message": "Internet connection lost. Switching to offline mode.",
                "recoverable": True,
                "original_error": str(e)
            }
        }
        
    except PermissionError as e:
        logger.error(f"STT Permission error: {e}")
        return {
            "status": "error", 
            "error": {
                "type": "permission",
                "message": "Microphone access denied. Please check your settings.",
                "recoverable": False,
                "original_error": str(e)
            }
        }
        
    except Exception as e:
        logger.exception(f"STT Unknown error: {e}")
        return {
            "status": "error",
            "error": {
                "type": "unknown", 
                "message": f"Unexpected error: {str(e)}",
                "recoverable": False,
                "original_error": str(e)
            }
        }