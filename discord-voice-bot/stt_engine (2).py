import os
import asyncio
import urllib.request
import urllib.parse
import json


class DeepgramSTT:
    """
    Deepgram Speech-to-Text engine.

    Uses Deepgram's current REST API directly instead of the old
    deepgram.listen.rest.ListenV1Rest SDK path.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is missing. Add it to your .env file."
            )

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe recorded audio bytes with Deepgram.

        The Discord audio sink should provide WAV/PCM-compatible audio.
        """

        if not audio_data:
            return ""

        return await asyncio.to_thread(self._transcribe_sync, audio_data)

    def _transcribe_sync(self, audio_data: bytes) -> str:
        # Deepgram Listen API endpoint.
        params = urllib.parse.urlencode({
            "model": "nova-3",
            "smart_format": "true",
            "punctuate": "true",
        })

        url = f"https://api.deepgram.com/v1/listen?{params}"

        # Most Discord recording sinks provide WAV/container audio.
        # If the data is raw PCM, the API can still work when the sink
        # already supplies a compatible WAV/PCM payload.
        content_type = self._detect_content_type(audio_data)

        request = urllib.request.Request(
            url,
            data=audio_data,
            method="POST",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": content_type,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()

            result = json.loads(raw.decode("utf-8"))

            transcript = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )

            return transcript.strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Deepgram API error {e.code}: {error_body}"
            ) from e

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not connect to Deepgram: {e.reason}"
            ) from e

        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Deepgram returned an invalid response."
            ) from e

    @staticmethod
    def _detect_content_type(audio_data: bytes) -> str:
        """
        Detect common audio containers from their magic bytes.
        """
        # WAV / RIFF
        if audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
            return "audio/wav"

        # OGG / Opus
        if audio_data[:4] == b"OggS":
            return "audio/ogg"

        # WebM / Matroska
        if audio_data[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm"

        # MP3 with ID3 header
        if audio_data[:3] == b"ID3":
            return "audio/mpeg"

        # MP3 frame sync
        if len(audio_data) >= 2 and audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0:
            return "audio/mpeg"

        # Default used by the existing bot for PCM-style audio.
        return "audio/wav"
