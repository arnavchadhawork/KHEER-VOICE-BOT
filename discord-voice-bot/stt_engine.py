"""
Deepgram Speech-to-Text Engine
Optimized for Hindi and Hinglish voice recognition.
Accepts raw 16-bit PCM audio from the Discord AudioSink.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("voice-bot.stt")


class DeepgramSTT:
    """Speech-to-Text using Deepgram's REST API."""

    def __init__(self, api_key: str | None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is missing. Put it in your .env file."
            )

    async def transcribe(self, audio_data: bytes) -> str | None:
        """Transcribe raw 16-bit PCM, 48 kHz, stereo audio."""
        if not audio_data or len(audio_data) < 8000:
            return None

        try:
            return await asyncio.to_thread(self._transcribe_sync, audio_data)
        except Exception as e:
            log.error(f"Deepgram STT error: {e}", exc_info=True)
            return None

    def _transcribe_sync(self, audio_data: bytes) -> str | None:
        import io
        import wave
        import audioop

        # Downmix 48kHz stereo PCM to clean mono PCM (STT accuracy is much higher on mono)
        try:
            mono_pcm = audioop.tomono(audio_data, 2, 0.5, 0.5)
        except Exception:
            mono_pcm = audio_data

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(mono_pcm)

        wav_bytes = wav_io.getvalue()

        # Save debug recording
        try:
            with open("last_recording.wav", "wb") as f:
                f.write(wav_bytes)
        except Exception:
            pass

        # 1. Primary: Hindi transcription (nova-2 language=hi)
        transcript = self._call_deepgram(wav_bytes, language="hi")
        if transcript:
            return transcript

        # 2. Fallback: Auto-detect language (catches Hinglish / English phrases)
        return self._call_deepgram(wav_bytes, detect_language=True)

    def _call_deepgram(
        self,
        wav_bytes: bytes,
        language: str | None = None,
        detect_language: bool = False,
    ) -> str | None:
        query = {
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true",
        }
        if language:
            query["language"] = language
        if detect_language:
            query["detect_language"] = "true"

        url = f"https://api.deepgram.com/v1/listen?{urllib.parse.urlencode(query)}"

        request = urllib.request.Request(
            url,
            data=wav_bytes,
            method="POST",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            transcript = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            ).strip()

            if transcript:
                log.info(f"🎤 Deepgram recognized: \"{transcript}\"")
                return transcript

            log.info(f"ℹ️ Deepgram ({'hi' if language else 'auto-detect'}): No words detected in audio segment.")
            return None

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            log.error(f"Deepgram HTTP {e.code}: {body}")
            return None

        except Exception as e:
            log.error(f"Deepgram connection error: {e}")
            return None
