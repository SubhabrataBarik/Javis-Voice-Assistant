import logging
# import threading
# import time
import asyncio
from typing import Type
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
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))  # Fixed: use 0.5, not 3

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
            return obj.get(name, default)  # type: ignore
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
    # prefer explicit utterance if present
    utter = _get_attr(event, "utterance", None)
    if utter:
        return utter
    # next, finalized words
    utter = _construct_utterance_from_words(event)
    if utter:
        return utter
    # fallback to transcript (may be unformatted)
    return _get_attr(event, "transcript", None)

# -------------------------------
# The STTService Class
# -------------------------------
class STTService:
    """
    Class-based STT service for better organization and reusability
    """
    
    def __init__(self, api_key: str, default_timeout: float = 30.0, vad_threshold: float = 0.5):
        self.api_key = api_key
        self.default_timeout = default_timeout
        self.vad_threshold = vad_threshold
        
    def capture_speech(self, timeout: float = None) -> str | None:
        """
        Capture one finalized utterance from real-time speech input
        
        Args:
            timeout: Maximum time to wait for speech (uses default if None)
            
        Returns:
            The captured utterance text, or None if no speech captured
        """
        if timeout is None:
            timeout = self.default_timeout
            
        # Shared structures / synchronization
        result = {"utter": None}
        result_lock = threading.Lock()
        stop_event = threading.Event()

        # Keep a small pending store in case formatted event arrives a tiny bit later
        pending_unformatted = {}
        pending_lock = threading.Lock()
        # Timer dict to cancel timers if formatted arrives
        pending_timers = {}

        # -------------------------------
        # Callbacks
        # -------------------------------
        def on_begin(client: Type[StreamingClient], event: BeginEvent):
            logger.info(f"🔹 STT session started: {event.id}")
            print("🎙️ Listening... Speak into your mic")

        def _set_result_and_stop(utter_text: str):
            with result_lock:
                # set only if not set yet
                if not result["utter"]:
                    result["utter"] = utter_text
                    stop_event.set()

        def _pending_timeout_handler(turn_order):
            # If no formatted arrived after grace period, accept pending unformatted
            with pending_lock:
                utter = pending_unformatted.pop(turn_order, None)
                timer = pending_timers.pop(turn_order, None)
            if utter:
                _set_result_and_stop(utter)

        def on_turn(client: Type[StreamingClient], event: TurnEvent):
            """
            We prefer formatted + end_of_turn events. If we get an unformatted end_of_turn,
            store it temporarily and wait a small grace interval for a formatted version.
            """
            turn_order = _get_attr(event, "turn_order", None)
            utter = _get_best_utter(event)
            end_of_turn = _get_attr(event, "end_of_turn", False)
            is_formatted = _get_attr(event, "turn_is_formatted", False)

            # Print live intermediate info (useful while debugging)
            if utter:
                status = "end_of_turn=True" if end_of_turn else "end_of_turn=False"
                fmt = "formatted" if is_formatted else "unformatted"
                print(f"🗣️ {utter} ({status}, {fmt})")

            # If formatted & end_of_turn -> take immediately
            if end_of_turn and is_formatted and utter:
                _set_result_and_stop(utter)
                return

            # If end_of_turn & unformatted -> store temporarily
            if end_of_turn and utter:
                if turn_order is None:
                    # no turn ordering available: accept this utter immediately
                    _set_result_and_stop(utter)
                    return

                with pending_lock:
                    pending_unformatted[turn_order] = utter
                    # if an old timer exists for this turn, cancel it
                    t = pending_timers.pop(turn_order, None)
                    if t:
                        t.cancel()
                    # create a small grace timer for formatted replacement (e.g., 0.45s)
                    timer = threading.Timer(0.45, _pending_timeout_handler, args=(turn_order,))
                    timer.daemon = True
                    pending_timers[turn_order] = timer
                    timer.start()
                return

            # Otherwise (partial/not end_of_turn) -> do nothing here
            return

        def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
            # If nothing captured yet, but pending unformatted exist, use the earliest one
            with pending_lock:
                pending_items = sorted(pending_unformatted.items(), key=lambda x: x[0])
                pending_unformatted.clear()
                # cancel pending timers
                for t in pending_timers.values():
                    try:
                        t.cancel()
                    except Exception:
                        pass
                pending_timers.clear()

            if pending_items and not result["utter"]:
                # take first pending if nothing else captured
                _set_result_and_stop(pending_items[0][1])

        def on_error(client: Type[StreamingClient], error: StreamingError):
            logger.error(f"STT error: {error}")
            # wake up the main thread to avoid hanging
            stop_event.set()

        # -------------------------------
        # Create client and register callbacks
        # -------------------------------
        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
                open_timeout=30  # increase handshake timeout here
            )
        )
        
        # Register callbacks on the single client instance
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)

        # Connect and start stream
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                speech_model="universal-streaming-english",
                vad_threshold=self.vad_threshold
            )
        )

        stream_exc = {"error": None}
        def _stream_thread_fn():
            try:
                client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
            except Exception as e:
                stream_exc["error"] = e
                stop_event.set()

        stream_thread = threading.Thread(target=_stream_thread_fn, daemon=True)
        stream_thread.start()

        # Wait until we get a result or timeout
        waited = 0.0
        poll_interval = 0.05 # Interval in seconds for checking if speech completed
        while not stop_event.is_set() and waited < timeout:
            time.sleep(poll_interval)
            waited += poll_interval

        # If we timed out or got a result, ensure stream stops
        try:
            client.disconnect(terminate=True)
        except Exception:
            pass

        # Join stream thread
        stream_thread.join(timeout=2.0)

        # If stream had exception, raise it so caller can handle/log
        if stream_exc.get("error"):
            raise stream_exc["error"]

        # Return the final utterance (if any)
        with result_lock:
            return result["utter"]

# -------------------------------
# Initialize the service instance (GLOBAL)
# -------------------------------
stt_service_instance = STTService(
    api_key=API_KEY,
    default_timeout=DEFAULT_STT_TIMEOUT,
    vad_threshold=VAD_THRESHOLD
)

# -------------------------------
# Simple wrapper function for LangGraph
# -------------------------------
def stt_service(state: VoiceState) -> dict:
    """
    Simple wrapper function for LangGraph nodes - calls the class method.
    This keeps LangGraph integration clean while using the organized class structure.
    """
    try:
        # Use the class instance to capture speech
        utter_final = stt_service_instance.capture_speech()
        
        if not utter_final:
            # No speech captured within timeout
            logger.info("No speech captured within timeout.")
            return {}  # Return empty dict, don't modify state

        print("\n🟢 Final utterance captured:")
        print(utter_final)

        # Get existing messages or create empty list
        current_messages = getattr(state, "messages", []) or []
        
        # Add new message
        updated_messages = current_messages + [HumanMessage(content=utter_final)]
        
        # Return dict to update state
        return {"messages": updated_messages}
        
    except Exception as e:
        logger.exception("Failed to process utterance: %s", e)
        return {}  # Return empty dict if failed
