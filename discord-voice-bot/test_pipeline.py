import sys
import os
import asyncio
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import subprocess
from dotenv import load_dotenv

load_dotenv()

async def test_audio_pipeline():
    from tts_engine import ElevenLabsTTS
    from stt_engine import DeepgramSTT

    tts = ElevenLabsTTS(os.getenv('ELEVENLABS_API_KEY'), os.getenv('ELEVENLABS_VOICE_ID'))
    stt = DeepgramSTT(os.getenv('DEEPGRAM_API_KEY'))

    print("1. Synthesizing Hindi test phrase via ElevenLabs...")
    mp3_bytes = await tts.synthesize("नमस्ते खीर, तुम कैसी हो? आज क्या कर रही हो?")
    print(f"Generated MP3: {len(mp3_bytes)} bytes")

    print("2. Converting MP3 to 48kHz stereo raw PCM (simulating Discord input)...")
    proc = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '48000', '-ac', '2', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    pcm_bytes, _ = proc.communicate(input=mp3_bytes)
    print(f"PCM size: {len(pcm_bytes)} bytes")

    print("3. Transcribing with DeepgramSTT...")
    transcript = await stt.transcribe(pcm_bytes)
    print(f"🎤 TRANSCRIPTION RESULT: '{transcript}'")

if __name__ == '__main__':
    asyncio.run(test_audio_pipeline())
