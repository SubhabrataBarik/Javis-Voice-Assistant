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




# import logging
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
# # Event callbacks
# # -------------------------------
# def on_begin(self: Type[StreamingClient], event: BeginEvent):
#     print(f"🔹 Session started: {event.id}")

# def on_turn(self: Type[StreamingClient], event: TurnEvent):
#     # Print the transcript as it arrives
#     print(f"\n🗣️ {event.transcript} (end_of_turn={event.end_of_turn})")

#     # Optional: auto-format turns
#     if event.end_of_turn and not event.turn_is_formatted:
#         params = StreamingSessionParameters(format_turns=True)
#         self.set_params(params)

# def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
#     print(f"✅ Session terminated: {event.audio_duration_seconds} seconds of audio processed")

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
#             format_turns=True,  # Optional: gets formatted text with punctuation
#             speech_model="universal-streaming-english"
#         )
#     )

#     try:
#         # Stream microphone audio
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
    StreamingSessionParameters
)

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Missing AssemblyAI API key. Check your .env file.")

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Store final transcript
# -------------------------------
final_text = []
seen_words = set()  # Track already added words

# -------------------------------
# Event callbacks
# -------------------------------
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"🔹 Session started: {event.id}")
    print("🎙️ Listening... Speak into your mic")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    # Extract only finalized words
    new_words = []
    for w in event.words:
        if w.word_is_final and w.text not in seen_words:
            new_words.append(w.text)
            seen_words.add(w.text)
    
    if new_words:
        final_text.extend(new_words)
        # Print live text as it comes
        print(f"🗣️ {' '.join(new_words)} (end_of_turn={event.end_of_turn})")
    
    # Optional: auto-format turns
    if event.end_of_turn and not event.turn_is_formatted:
        params = StreamingSessionParameters(format_turns=True)
        self.set_params(params)

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"\n✅ Session terminated: {event.audio_duration_seconds} seconds of audio processed")
    print("\n💬 Final full transcript for LLM:")
    llm_input = " ".join(final_text)
    print(llm_input)

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"❌ Error occurred: {error}")

# -------------------------------
# Main streaming function
# -------------------------------
def main():
    client = StreamingClient(
        StreamingClientOptions(
            api_key=API_KEY,
            api_host="streaming.assemblyai.com"
        )
    )

    # Register callbacks
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # Connect and start streaming from mic
    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,  # Adds punctuation and casing
            speech_model="universal-streaming-english"
        )
    )

    try:
        # Stream microphone audio
        client.stream(
            aai.extras.MicrophoneStream(sample_rate=16000)
        )
    finally:
        # Gracefully disconnect
        client.disconnect(terminate=True)

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()
