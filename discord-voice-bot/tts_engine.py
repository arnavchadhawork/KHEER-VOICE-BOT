"""
ElevenLabs Text-to-Speech Engine
Converts text responses to realistic girl voice audio.
"""

import logging
import asyncio
from elevenlabs import ElevenLabs

log = logging.getLogger("voice-bot.tts")


class ElevenLabsTTS:
    """Text-to-Speech using ElevenLabs API."""

    def __init__(self, api_key: str, voice_id: str):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id

    async def synthesize(self, text: str) -> bytes | None:
        """
        Convert text to speech audio bytes.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes (MP3 format) or None if failed
        """
        try:
            # Run the sync API in an executor
            audio_generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",  # Best for Hindi/Hinglish
                    output_format="mp3_44100_128",
                    voice_settings={
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.4,           # Some expressiveness
                        "use_speaker_boost": True,
                    },
                ),
            )

            # Collect all chunks
            audio_bytes = b""
            for chunk in audio_generator:
                audio_bytes += chunk

            if audio_bytes:
                log.info(f"TTS generated {len(audio_bytes)} bytes of audio")
                return audio_bytes

            return None

        except Exception as e:
            log.error(f"ElevenLabs TTS error: {e}", exc_info=True)
            return None
