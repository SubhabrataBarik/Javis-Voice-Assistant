# import logging
# import threading
# from typing import Type
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
#     StreamingSessionParameters
# )

# # -------------------------------
# # Load environment variables
# # -------------------------------
# load_dotenv()
# API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# if not API_KEY:
#     raise ValueError("❌ Missing AssemblyAI API key. Check your .env file.")

# # -------------------------------
# # Logging
# # -------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # -------------------------------
# # Thread-safe storage
# # -------------------------------
# final_text = []                        # list of finalized utterances (in order)
# final_text_lock = threading.Lock()

# pending_turns = {}                     # turn_order -> utter (unformatted or initial)
# pending_lock = threading.Lock()

# appended_turns = set()                 # set of turn_order values already appended

# # -------------------------------
# # Helpers
# # -------------------------------
# def _get_attr(obj, name, default=None):
#     try:
#         return getattr(obj, name, default)
#     except Exception:
#         # sometimes SDK gives dict-like objects
#         try:
#             return obj.get(name, default)  # type: ignore
#         except Exception:
#             return default

# def _construct_utterance_from_words(event: TurnEvent) -> str | None:
#     words = _get_attr(event, "words", None)
#     if not words:
#         return None
#     final_words = []
#     for w in words:
#         # support both attribute and dict access
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
#     # Prefer event.utterance if present
#     utter = _get_attr(event, "utterance", None)
#     if utter:
#         return utter
#     # Then prefer constructed finalized words
#     utter = _construct_utterance_from_words(event)
#     if utter:
#         return utter
#     # Finally fallback to event.transcript
#     return _get_attr(event, "transcript", None)

# # -------------------------------
# # Event callbacks
# # -------------------------------
# def on_begin(self: Type[StreamingClient], event: BeginEvent):
#     print(f"🔹 Session started: {event.id}")
#     print("🎙️ Listening... Speak into your mic")

# def on_turn(self: Type[StreamingClient], event: TurnEvent):
#     """
#     Handle each TurnEvent and ensure we only append one final utterance per turn_order.
#     Prefer formatted utterance (turn_is_formatted=True). If formatted hasn't arrived yet,
#     store unformatted in pending_turns and replace it if formatted arrives later.
#     """
#     turn_order = _get_attr(event, "turn_order", None)
#     if turn_order is None:
#         # If no turn_order, treat as a normal utter and append once when end_of_turn
#         utter = _get_best_utter(event)
#         if _get_attr(event, "end_of_turn", False) and utter:
#             with final_text_lock:
#                 final_text.append(utter)
#                 print(f"🗣️ {utter} (end_of_turn=True)")
#         else:
#             # print partial live text (if any)
#             if utter:
#                 print(f"🗣️ {utter} (partial)")
#         return

#     # Determine utterance for this event
#     utter = _get_best_utter(event)
#     end_of_turn = _get_attr(event, "end_of_turn", False)
#     is_formatted = _get_attr(event, "turn_is_formatted", False)

#     # Print live (show whatever we have)
#     if utter:
#         status = "end_of_turn=True" if end_of_turn else "end_of_turn=False"
#         fmt = "formatted" if is_formatted else "unformatted"
#         print(f"🗣️ {utter} ({status}, {fmt})")

#     # If we've already appended this turn, ignore
#     with final_text_lock:
#         if turn_order in appended_turns:
#             return

#     # If this event is formatted and end_of_turn => append immediately (preferred)
#     if end_of_turn and is_formatted and utter:
#         with final_text_lock:
#             final_text.append(utter)
#             appended_turns.add(turn_order)
#         # ensure we remove any pending unformatted entry
#         with pending_lock:
#             pending_turns.pop(turn_order, None)
#         return

#     # If end_of_turn and not formatted: store in pending (may be replaced by formatted later)
#     if end_of_turn and utter:
#         with pending_lock:
#             pending_turns[turn_order] = utter
#         return

#     # If not end_of_turn: do nothing (we wait until end_of_turn)
#     return

# def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
#     print(f"\n✅ Session terminated: {event.audio_duration_seconds} seconds of audio processed")
#     # Append any pending turns that never received a formatted version
#     with pending_lock:
#         pending_items = sorted(pending_turns.items(), key=lambda x: x[0])  # sort by turn_order
#     with final_text_lock:
#         for turn_order, utter in pending_items:
#             if turn_order not in appended_turns:
#                 final_text.append(utter)
#                 appended_turns.add(turn_order)
#         # clear pending
#         with pending_lock:
#             pending_turns.clear()

#     # Print final transcript: one utterance per line (formatted when available)
#     print("\n💬 Final full transcript for LLM:")
#     with final_text_lock:
#         if final_text:
#             print("\n".join(final_text))
#         else:
#             print("[no transcript captured]")

# def on_error(self: Type[StreamingClient], error: StreamingError):
#     print(f"❌ Error occurred: {error}")

# # -------------------------------
# # Main streaming function
# # -------------------------------
# def main():
#     client = StreamingClient(
#         StreamingClientOptions(
#             api_key=API_KEY,
#             api_host="streaming.assemblyai.com"
#         )
#     )

#     # Register callbacks
#     client.on(StreamingEvents.Begin, on_begin)
#     client.on(StreamingEvents.Turn, on_turn)
#     client.on(StreamingEvents.Termination, on_terminated)
#     client.on(StreamingEvents.Error, on_error)

#     # Connect and start streaming from mic
#     client.connect(
#         StreamingParameters(
#             sample_rate=16000,
#             format_turns=True,  # request formatted turns (punctuation & casing)
#             speech_model="universal-streaming-english"
#         )
#     )

#     try:
#         # Stream microphone audio (uses PyAudio or sounddevice depending on aai.extras)
#         client.stream(
#             aai.extras.MicrophoneStream(sample_rate=16000)
#         )
#     finally:
#         # Gracefully disconnect
#         client.disconnect(terminate=True)

# # -------------------------------
# # Entry point
# # -------------------------------
# if __name__ == "__main__":
#     main()






















import logging
import threading
import time
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
    StreamingSessionParameters,
)

# For inserting the final utterance into the VoiceState
from langchain_core.messages import HumanMessage
from VoiceState import VoiceState

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
# seconds to wait before giving up (safety)
DEFAULT_STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "30"))

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
# The main service function to be called by your StateGraph
# -------------------------------
def stt_service(state: VoiceState, timeout: float | None = None) -> VoiceState:
    """
    Start a short real-time STT session, capture one finalized utterance, append it
    to state.messages as a HumanMessage, and return the updated state.

    - Prefers formatted turn (if AssemblyAI returns it).
    - Falls back to unformatted finalized words or transcript.
    - Stops after capturing one final utterance or after `timeout` seconds.
    """
    if timeout is None:
        timeout = DEFAULT_STT_TIMEOUT

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
        StreamingClientOptions(api_key=API_KEY, api_host="streaming.assemblyai.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # Connect and start stream in a thread (stream blocks)
    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
            speech_model="universal-streaming-english",
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
    poll_interval = 0.05
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

    # Final utterance (if any)
    with result_lock:
        utter_final = result["utter"]

    if not utter_final:
        # No speech captured within timeout
        logger.info("No speech captured within timeout.")
        return state

    # Append as HumanMessage to VoiceState messages so llm_service can pick it up
    try:
        # ensure state.messages exists and is a list
        messages = getattr(state, "messages", None)
        if messages is None:
            # If VoiceState expects a different structure, adapt here
            state.messages = []
            messages = state.messages

        if isinstance(messages, list):
            messages.append(HumanMessage(content=utter_final))
        else:
            # fallback: try to call an `add_message` method if available
            add_fn = getattr(state, "add_message", None)
            if callable(add_fn):
                add_fn(HumanMessage(content=utter_final))
            else:
                # unknown state structure — attach messages list anyway
                state.messages = [HumanMessage(content=utter_final)]
    except Exception as e:
        logger.exception("Failed to append utterance to state.messages: %s", e)
        # still return state even if storing failed

    # also print final utterance for clarity
    print("\n🟢 Final utterance captured:")
    print(utter_final)

    return state
























# from VoiceState import VoiceState
# import os
# import assemblyai as aai
# from dotenv import load_dotenv
# load_dotenv()

# # Set API key from environment
# aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# # Audio source
# audio_file = r"D:\PENDRIVE 32 GB\Game\AI_PROJECTS\javis\20230607_me_canadian_wildfires.mp3"

# # Transcription config
# config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
# transcriber = aai.Transcriber(config=config)
# transcript = transcriber.transcribe(audio_file)

# if transcript.status == "error":
#     raise RuntimeError(f"Transcription failed: {transcript.error}")

# print(transcript.text)