import logging
import asyncio
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
# Clean Async STT Service with asyncio.Event() and asyncio.Queue()
# -------------------------------
class CleanAsyncSTTService:
    """
    Clean async STT service using asyncio.Event() and asyncio.Queue()
    for state management - no threading complexity!
    """
    
    def __init__(self, api_key: str, default_timeout: float = 30.0, vad_threshold: float = 0.5):
        self.api_key = api_key
        self.default_timeout = default_timeout
        self.vad_threshold = vad_threshold
        
        # External callback handlers (clean separation)
        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None
        self.on_session_start: Optional[Callable[[str], None]] = None
        
    async def capture_speech(self, timeout: float = None) -> str | None:
        """
        Clean async capture using asyncio.Event() and asyncio.Queue()
        """
        if timeout is None:
            timeout = self.default_timeout
            
        # ✅ Clean async state management - no locks needed!
        partial_queue = asyncio.Queue()          # For partial results
        final_queue = asyncio.Queue()            # For final results  
        completion_event = asyncio.Event()       # Session completion
        error_event = asyncio.Event()            # Error signaling
        
        # Get current event loop for thread-safe signaling
        loop = asyncio.get_running_loop()
        
        # Store for pending formatted results
        pending_results = {}
        
        # -------------------------------
        # Clean callback handlers
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
            """Clean turn handling with async queues"""
            turn_order = _get_attr(event, "turn_order", None)
            utter = _get_best_utter(event)
            end_of_turn = _get_attr(event, "end_of_turn", False)
            is_formatted = _get_attr(event, "turn_is_formatted", False)

            if not utter:
                return

            # Print live feedback
            status = "end_of_turn=True" if end_of_turn else "end_of_turn=False"
            fmt = "formatted" if is_formatted else "unformatted"
            print(f"🗣️ {utter} ({status}, {fmt})")

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

            # Handle end of turn
            if end_of_turn:
                if is_formatted:
                    # Formatted result - use immediately
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(final_queue.put(utter))
                    )
                    loop.call_soon_threadsafe(completion_event.set)
                else:
                    # Unformatted result - store and wait for formatted version
                    if turn_order is not None:
                        pending_results[turn_order] = utter
                        # Schedule timeout for this result
                        loop.call_later(
                            0.45, 
                            lambda: self._handle_pending_timeout(turn_order, pending_results, final_queue, completion_event)
                        )
                    else:
                        # No turn order - use unformatted immediately  
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(final_queue.put(utter))
                        )
                        loop.call_soon_threadsafe(completion_event.set)

        def on_terminated(client, event: TerminationEvent):
            """Clean termination handling"""
            logger.info("🔹 STT session terminated")
            
            # Use any pending results if no final result yet
            if pending_results and not completion_event.is_set():
                # Get earliest pending result
                earliest_order = min(pending_results.keys()) if pending_results else None
                if earliest_order is not None:
                    utter = pending_results[earliest_order]
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(final_queue.put(utter))
                    )
            
            # Signal completion
            loop.call_soon_threadsafe(completion_event.set)

        def on_error(client, error: StreamingError):
            """Clean error handling"""
            logger.error(f"STT error: {error}")
            loop.call_soon_threadsafe(error_event.set)
            loop.call_soon_threadsafe(completion_event.set)

        # -------------------------------
        # Setup and run streaming
        # -------------------------------
        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
                open_timeout=30
            )
        )
        
        # Register clean callbacks
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated) 
        client.on(StreamingEvents.Error, on_error)

        # Connect
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                speech_model="universal-streaming-english",
                vad_threshold=self.vad_threshold
            )
        )

        # ✅ Clean async execution with proper resource management
        stream_task = None
        final_result = None
        
        try:
            # Start streaming task
            stream_task = asyncio.create_task(
                asyncio.to_thread(client.stream, aai.extras.MicrophoneStream(sample_rate=16000))
            )
            
            # Start partial result processor
            partial_task = asyncio.create_task(self._process_partial_results(partial_queue))
            
            # Wait for final result with timeout
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
            
            # Cancel tasks
            if stream_task and not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                    
            # Cancel partial processor
            partial_task.cancel()
            try:
                await partial_task
            except asyncio.CancelledError:
                pass

        return final_result
    
    def _handle_pending_timeout(self, turn_order: int, pending_results: dict, 
                              final_queue: asyncio.Queue, completion_event: asyncio.Event):
        """Handle timeout for pending unformatted results"""
        if turn_order in pending_results and not completion_event.is_set():
            utter = pending_results.pop(turn_order)
            asyncio.create_task(final_queue.put(utter))
            completion_event.set()
    
    async def _process_partial_results(self, partial_queue: asyncio.Queue):
        """Process partial results as they arrive"""
        try:
            while True:
                # Non-blocking check for partial results
                try:
                    partial = await asyncio.wait_for(partial_queue.get(), timeout=0.1)
                    # Could do something with partial results here
                    # e.g., update UI in real-time
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

# -------------------------------
# Service instance with external callbacks
# -------------------------------
stt_service_instance = CleanAsyncSTTService(
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
# Clean async wrapper for LangGraph
# -------------------------------
async def stt_service(state: VoiceState) -> dict:
    """
    Clean async wrapper using asyncio.Event() and asyncio.Queue()
    """
    try:
        # ✅ Clean async call - no locks, no threading complexity
        utter_final = await stt_service_instance.capture_speech()
        
        if not utter_final:
            logger.info("No speech captured within timeout.")
            return {}

        # Update state cleanly
        current_messages = getattr(state, "messages", []) or []
        updated_messages = current_messages + [HumanMessage(content=utter_final)]
        
        return {"messages": updated_messages}
        
    except Exception as e:
        logger.exception("Failed to process utterance: %s", e)
        return {}

# -------------------------------
# Enhanced version with real-time features
# -------------------------------
class EnhancedAsyncSTTService(CleanAsyncSTTService):
    """
    Enhanced version with real-time features and better state management
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_transcript = ""
        self.transcript_queue = asyncio.Queue()
    
    async def capture_speech_with_live_updates(self, timeout: float = None) -> str | None:
        """
        Enhanced version that provides live transcript updates
        """
        # Setup live update callback
        original_partial = self.on_partial
        
        async def live_partial_handler(text: str):
            self.live_transcript = text
            await self.transcript_queue.put(text)
            if original_partial:
                original_partial(text)
        
        # Temporarily override partial handler
        self.on_partial = live_partial_handler
        
        try:
            # Run capture with live updates
            result = await self.capture_speech(timeout)
            return result
        finally:
            # Restore original handler
            self.on_partial = original_partial
    
    async def get_live_transcript(self) -> str:
        """Get the current live transcript"""
        return self.live_transcript

# Example usage for enhanced features
enhanced_stt_service = EnhancedAsyncSTTService(
    api_key=API_KEY,
    default_timeout=DEFAULT_STT_TIMEOUT, 
    vad_threshold=VAD_THRESHOLD
)
