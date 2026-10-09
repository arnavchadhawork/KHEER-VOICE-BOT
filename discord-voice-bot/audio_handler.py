"""
Audio handler for discord-ext-voice-recv.

Receives decoded PCM audio from Discord continuously,
detects user speech and silence via energy-based Voice Activity Detection (VAD),
preserves pre-roll & post-roll padding so no words are clipped,
and provides clean mixed audio buffers.
"""

import audioop
import collections
import logging
import struct
import threading
import time

from discord.ext import voice_recv

log = logging.getLogger("voice-bot.audio")

# Threshold in 16-bit PCM amplitude (-32768 to 32767).
# 150 is low enough to catch soft Hindi vowels, whispers, and normal speech,
# while high enough to reject digital silence (which is near 0).
VOICE_ENERGY_THRESHOLD = 150


class AudioSink(voice_recv.AudioSink):
    """Collect PCM audio continuously with smooth voice activity & silence tracking."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.audio_data: dict[int, bytearray] = {}
        # Pre-roll ring buffer to save ~200ms before speech starts (10 packets of 20ms)
        self.preroll: dict[int, collections.deque] = {}
        self.last_speech_time: float = 0.0
        self.has_speech: bool = False
        self.paused: bool = False

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        """Called by discord-ext-voice-recv for every received 20ms voice packet."""
        if self.paused or not data or not data.pcm:
            return

        pcm = data.pcm
        try:
            peak = audioop.max(pcm, 2)
        except Exception:
            peak = 0

        user_id = getattr(user, "id", None) or getattr(data.packet, "ssrc", 0)

        with self.lock:
            if user_id not in self.preroll:
                self.preroll[user_id] = collections.deque(maxlen=10)

            if peak > VOICE_ENERGY_THRESHOLD:
                # Active speech detected
                if user_id not in self.audio_data:
                    self.audio_data[user_id] = bytearray()
                    # Prepend pre-roll buffer so the start of the word isn't clipped
                    for prev_pcm in self.preroll[user_id]:
                        self.audio_data[user_id].extend(prev_pcm)

                self.audio_data[user_id].extend(pcm)
                self.last_speech_time = time.time()
                self.has_speech = True

            else:
                # Below threshold
                if self.has_speech:
                    # Trailing voice or natural pause inside a sentence
                    if user_id in self.audio_data:
                        self.audio_data[user_id].extend(pcm)
                else:
                    # Silence before speech: store in rolling pre-buffer
                    self.preroll[user_id].append(pcm)

    def get_speech_duration(self) -> float:
        """Returns approximate seconds of recorded audio for the most active speaker."""
        with self.lock:
            if not self.audio_data or not self.has_speech:
                return 0.0
            max_bytes = max(len(buf) for buf in self.audio_data.values())
            # 48000 Hz, 2 channels, 16-bit (2 bytes per sample) = 192000 bytes per second
            return max_bytes / 192000.0

    def get_silence_duration(self) -> float:
        """Returns seconds elapsed since active speech was last detected."""
        with self.lock:
            if not self.has_speech or self.last_speech_time == 0.0:
                return 0.0
            return time.time() - self.last_speech_time

    def get_and_clear_audio(self) -> bytes | None:
        """Return all users' PCM audio mixed into one 16-bit PCM buffer and reset."""
        with self.lock:
            if not self.audio_data or not self.has_speech:
                return None

            buffers = [bytes(buf) for buf in self.audio_data.values() if len(buf) > 0]
            self.audio_data.clear()
            self.has_speech = False
            self.last_speech_time = 0.0

        if not buffers:
            return None

        if len(buffers) == 1:
            return buffers[0]

        max_length = max(len(buf) for buf in buffers)
        max_length -= max_length % 2

        combined = bytearray(max_length)

        # Mix 16-bit signed PCM samples with clipping.
        for buf in buffers:
            usable_length = min(len(buf), max_length)
            usable_length -= usable_length % 2

            for i in range(0, usable_length, 2):
                existing = struct.unpack_from("<h", combined, i)[0]
                new_sample = struct.unpack_from("<h", buf, i)[0]
                mixed = max(-32768, min(32767, existing + new_sample))
                struct.pack_into("<h", combined, i, mixed)

        return bytes(combined)

    def cleanup(self):
        """Release stored audio buffers."""
        with self.lock:
            self.audio_data.clear()
            self.preroll.clear()
            self.has_speech = False
            self.last_speech_time = 0.0