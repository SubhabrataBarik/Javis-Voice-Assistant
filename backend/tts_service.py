from VoiceState import VoiceState
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

# -------------------
# Setup ElevenLabs client
# -------------------
ffmpeg_path = r"D:\PENDRIVE 32 GB\Game\AI_PROJECTS\javis\ffmpeg-8.0-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

api_key = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api_key)


def tts_service(state: VoiceState):
    """
    TTS service node for LangGraph workflow.
    Converts the latest LLM message to speech and plays it.
    """
    # Get latest LLM message
    messages = state['messages'][-1].content

    # Convert text to audio bytes
    audio = client.text_to_speech.convert(
        text=messages,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Save audio temporarily (current approach)
    with open("audio.wav", "wb") as f:
        for chunk in audio:
            f.write(chunk)

    # Play audio
    with open("audio.wav", "rb") as f:
        play(f.read())

    return