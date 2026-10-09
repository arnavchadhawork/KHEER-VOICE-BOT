"""
🔥 Savage Girl Discord Voice Bot

Commands:
  !join   - Join the voice channel you are in
  !leave  - Leave voice channel
  !listen - Start receiving voice
  !stop   - Stop receiving voice
  !reset  - Reset conversation history

Uses:
  Discord voice receive -> Deepgram STT -> Gemini -> ElevenLabs TTS
"""

import os
import sys
import asyncio
import logging
import tempfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import discord
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

from stt_engine import DeepgramSTT
from llm_engine import GeminiLLM
from tts_engine import ElevenLabsTTS
from audio_handler import AudioSink

# ─── Safeguard: Prevent Opus packet errors from crashing PacketRouter ────────
import discord.ext.voice_recv.opus as _vr_opus
import discord.ext.voice_recv.router as _vr_router

def _patch_decode_packet(self, packet):
    assert self._decoder is not None
    if packet:
        try:
            pcm = self._decoder.decode(packet.decrypted_data, fec=False)
        except Exception:
            try:
                pcm = self._decoder.decode(None, fec=False)
            except Exception:
                pcm = b""
        return packet, pcm

    next_packet = self._buffer.peek_next()
    if next_packet is not None:
        try:
            pcm = self._decoder.decode(next_packet.decrypted_data, fec=True)
        except Exception:
            try:
                pcm = self._decoder.decode(None, fec=False)
            except Exception:
                pcm = b""
    else:
        try:
            pcm = self._decoder.decode(None, fec=False)
        except Exception:
            pcm = b""
    return packet, pcm

_vr_opus.PacketDecoder._decode_packet = _patch_decode_packet

def _patch_do_run(self):
    while not self._end_thread.is_set():
        self.waiter.wait()
        with self._lock:
            for decoder in list(self.waiter.items):
                try:
                    data = decoder.pop_data()
                    if data is not None and data.pcm:
                        self.sink.write(data.source, data)
                except Exception:
                    pass

_vr_router.PacketRouter._do_run = _patch_do_run

# ─── Setup ───────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("voice-bot")

# Filter out repetitive RTCP packet logs permanently
class RTCPFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Received unexpected rtcp packet" not in record.getMessage()

logging.getLogger("discord.ext.voice_recv.reader").addFilter(RTCPFilter())

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents)

# ─── Engines ─────────────────────────────────────────────────────────────────

stt = DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY"))

llm = GeminiLLM(
    api_key=os.getenv("GEMINI_API_KEY")
)

tts = ElevenLabsTTS(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
)

# guild_id -> active AudioSink
active_listeners: dict[int, AudioSink] = {}

# guild_id -> monitor task
listener_tasks: dict[int, asyncio.Task] = {}

# guild_id -> conversation history
conversation_history: dict[int, list[dict]] = {}

# guild_id -> response lock
response_locks: dict[int, asyncio.Lock] = {}


# ─── Events ──────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    log.info(
        f"✅ Bot is ready! Logged in as {bot.user} ({bot.user.id})"
    )
    log.info(f"🏠 In {len(bot.guilds)} server(s)")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="tumhari bakwas 💅",
        )
    )


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = (message.content or "").strip()
    channel_name = getattr(message.channel, "name", "DM")
    log.info(f"📩 [#{channel_name}] {message.author}: '{content}'")

    if not content:
        log.warning(
            "⚠️ Message content is empty! Make sure 'MESSAGE CONTENT INTENT' is enabled "
            "in https://discord.com/developers/applications under Bot -> Privileged Gateway Intents."
        )

    await bot.process_commands(message)


# ─── Diagnostic & Test Commands ─────────────────────────────────────────────

@bot.command(name="ping", help="Bot status aur ping check karo")
async def ping_cmd(ctx: commands.Context):
    latency = round(bot.latency * 1000)
    await ctx.send(f"Pong! `{latency}ms` 🏓 Kheer online hai aur sun rahi hai! 💅")


@bot.command(name="test", help="Sabhi AI engines ka live test run karo")
async def test_cmd(ctx: commands.Context):
    msg = await ctx.send("🔍 Running AI diagnostic tests...")

    # 1. Test LLM
    try:
        reply = await llm.generate_response([{"role": "user", "content": "hello"}])
        llm_status = f"✅ Gemini OK (`{reply}`)"
    except Exception as e:
        llm_status = f"❌ Gemini Error: `{e}`"

    # 2. Test TTS
    try:
        audio = await tts.synthesize("Haan main zinda hoon.")
        if audio and len(audio) > 5000:
            tts_status = f"✅ ElevenLabs OK (`{len(audio):,}` bytes)"
        else:
            tts_status = "❌ ElevenLabs: Empty audio"
    except Exception as e:
        tts_status = f"❌ ElevenLabs Error: `{e}`"

    # 3. Test STT
    try:
        await stt.transcribe(b"\x00" * 48000)
        stt_status = "✅ Deepgram STT OK"
    except Exception as e:
        stt_status = f"❌ Deepgram Error: `{e}`"

    in_vc = bool(ctx.voice_client and ctx.voice_client.is_connected())
    report = (
        "**🔥 Kheer Bot — Diagnostics Report**\n"
        f"• **Gemini LLM:** {llm_status}\n"
        f"• **ElevenLabs TTS:** {tts_status}\n"
        f"• **Deepgram STT:** {stt_status}\n"
        f"• **Voice Channel:** {'Connected ✅' if in_vc else 'Not Connected (type `!join`) ⚠️'}"
    )
    await msg.edit(content=report)


@bot.command(name="speak", help="Voice channel mein text bolwao: !speak <message>")
async def speak_cmd(ctx: commands.Context, *, text: str):
    if not ctx.voice_client or not ctx.voice_client.is_connected():
        await ctx.send("Pehle `!join` bol ke mujhe voice channel mein bula 🙄")
        return

    await ctx.send(f"Bol rahi hoon: *\"{text}\"* 🎤")
    audio = await tts.synthesize(text)
    if audio:
        await play_audio(ctx, audio)
    else:
        await ctx.send("Audio generate nahi ho payi 🤐")


@bot.command(name="talk", help="Kheer se chat/voice mein baat karo: !talk <message>")
async def talk_cmd(ctx: commands.Context, *, text: str):
    guild_id = ctx.guild.id
    history = conversation_history.setdefault(guild_id, [])
    history.append({"role": "user", "content": text})
    if len(history) > 20:
        conversation_history[guild_id] = history[-20:]

    reply = await llm.generate_response(conversation_history[guild_id])
    conversation_history[guild_id].append({"role": "assistant", "content": reply})

    await ctx.send(f"💬 **Kheer:** {reply}")

    if ctx.voice_client and ctx.voice_client.is_connected():
        audio = await tts.synthesize(reply)
        if audio:
            await play_audio(ctx, audio)


# ─── Voice Commands ─────────────────────────────────────────────────────────

@bot.command(name="join", help="Bot ko voice channel mein bulao")
async def join_vc(ctx: commands.Context):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send(
            "Pehle khud voice channel mein aa, phir mujhe bulana 🙄"
        )
        return

    channel = ctx.author.voice.channel

    if ctx.voice_client:
        if ctx.voice_client.is_connected():
            if ctx.voice_client.channel == channel:
                await ctx.send(
                    "Main toh pehle se yahan hoon, andhe ho kya? 👀"
                )
                return

            await ctx.voice_client.move_to(channel)
            await ctx.send(
                f"Chal aa gayi `{channel.name}` mein. Ab `!listen` bolo 💅"
            )
            return
        else:
            # Stale / dead connection cleanup
            try:
                await ctx.voice_client.disconnect(force=True)
            except Exception:
                pass

    try:
        # Connect with voice receive client
        await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=15.0, reconnect=True)
        await ctx.send(
            f"Aa gayi `{channel.name}` mein! "
            "`!listen` bolo sunna shuru karungi 🎤"
        )
    except (asyncio.TimeoutError, TimeoutError):
        log.error(f"Voice handshake timeout for channel {channel.name} ({channel.id})")
        if ctx.voice_client:
            try:
                await ctx.voice_client.disconnect(force=True)
            except Exception:
                pass
        await ctx.send(
            "⚠️ Voice connection timeout ho gaya!\n"
            "1. Check karo bot ke paas channel mein **Connect** aur **Speak** permission hai.\n"
            "2. Channel settings mein **Region Override** (e.g. Singapore / India) change karke dekho.\n"
            "3. Discord se bot ko right-click karke Disconnect karo aur phir `!join` bolo."
        )


@bot.command(name="leave", help="Bot ko voice channel se nikaalo")
async def leave_vc(ctx: commands.Context):
    if not ctx.voice_client:
        await ctx.send(
            "Main kisi voice channel mein hoon hi nahi 😒"
        )
        return

    guild_id = ctx.guild.id

    # Stop monitoring
    task = listener_tasks.pop(guild_id, None)
    if task:
        task.cancel()

    sink = active_listeners.pop(guild_id, None)
    if sink:
        sink.cleanup()

    conversation_history.pop(guild_id, None)

    vc = ctx.voice_client

    if isinstance(vc, voice_recv.VoiceRecvClient):
        try:
            if vc.is_listening():
                vc.stop_listening()
        except Exception:
            pass

    await vc.disconnect()
    await ctx.send("Chal bye 💋")


# ─── Listening ──────────────────────────────────────────────────────────────

@bot.command(name="listen", help="Bot sunna shuru karegi")
async def start_listening(ctx: commands.Context):
    vc = ctx.voice_client

    if not vc or not vc.is_connected():
        await ctx.send(
            "Pehle `!join` bol ke mujhe voice channel mein bula 🙄"
        )
        return

    if not isinstance(vc, voice_recv.VoiceRecvClient):
        await ctx.send(
            "Voice receive client nahi hai. `!leave` karke phir `!join` karo."
        )
        return

    guild_id = ctx.guild.id

    if guild_id in active_listeners:
        await ctx.send(
            "Sun toh rahi hoon already 😏"
        )
        return

    if guild_id not in conversation_history:
        conversation_history[guild_id] = []

    if guild_id not in response_locks:
        response_locks[guild_id] = asyncio.Lock()

    sink = AudioSink()
    active_listeners[guild_id] = sink

    try:
        if not vc.is_listening():
            vc.listen(sink)
            log.info(f"🎤 Continuous voice receive started in guild {guild_id}")
    except Exception as e:
        log.error(f"Failed to start voice receive: {e}", exc_info=True)
        active_listeners.pop(guild_id, None)
        await ctx.send(f"Voice receive error: `{e}`")
        return

    # Start speech monitoring loop
    listener_tasks[guild_id] = asyncio.create_task(voice_monitor_loop(ctx, sink))

    await ctx.send(
        "Bol ab, sun rahi hoon... 🎤💅"
    )


@bot.command(name="stop", help="Bot sunna band karegi")
async def stop_listening(ctx: commands.Context):
    guild_id = ctx.guild.id

    if guild_id not in active_listeners:
        await ctx.send(
            "Sun hi nahi rahi thi 😴"
        )
        return

    task = listener_tasks.pop(guild_id, None)
    if task:
        task.cancel()

    sink = active_listeners.pop(guild_id, None)
    if sink:
        sink.cleanup()

    vc = ctx.voice_client

    if isinstance(vc, voice_recv.VoiceRecvClient):
        try:
            if vc.is_listening():
                vc.stop_listening()
        except Exception as e:
            log.warning(f"Could not stop voice receive: {e}")

    await ctx.send(
        "Theek hai, sunna band kiya 😴"
    )


# ─── Speech Monitor Loop ───────────────────────────────────────────────────

async def voice_monitor_loop(ctx: commands.Context, sink: AudioSink):
    """
    Continuously monitors user speech without stopping/restarting voice receive.
    Detects when user finishes talking via silence pause (~1.2s), then processes.
    """
    guild_id = ctx.guild.id

    log.info(f"👂 Voice monitor loop active for guild {guild_id}")

    while guild_id in active_listeners:
        try:
            await asyncio.sleep(0.3)

            if guild_id not in active_listeners:
                break

            vc = ctx.voice_client
            if not vc or not vc.is_connected() or not isinstance(vc, voice_recv.VoiceRecvClient):
                break

            # If bot is talking or currently processing response, wait
            lock = response_locks.get(guild_id)
            if (lock and lock.locked()) or sink.paused or vc.is_playing():
                continue

            speech_sec = sink.get_speech_duration()
            silence_sec = sink.get_silence_duration()

            # Trigger condition:
            # User spoke for at least 0.3s AND paused for at least 0.9s
            # OR user spoke continuously for >10s
            if (speech_sec >= 0.3 and silence_sec >= 0.9) or speech_sec >= 10.0:
                sink.paused = True
                audio_bytes = sink.get_and_clear_audio()

                if audio_bytes and len(audio_bytes) > 8000:
                    log.info(f"🎤 Voice detected: {speech_sec:.1f}s audio. Processing...")
                    asyncio.create_task(handle_conversation(ctx, sink, audio_bytes))
                else:
                    sink.paused = False

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Error in voice monitor loop: {e}", exc_info=True)


# ─── Audio & Conversation Processing ────────────────────────────────────────

async def handle_conversation(
    ctx: commands.Context,
    sink: AudioSink,
    audio_bytes: bytes,
):
    """Process speech: Deepgram STT -> Gemini -> ElevenLabs TTS -> Playback."""
    guild_id = ctx.guild.id
    lock = response_locks.setdefault(guild_id, asyncio.Lock())

    async with lock:
        try:
            log.info("🎤 Converting speech to text (Deepgram)...")
            transcript = await stt.transcribe(audio_bytes)

            if not transcript or len(transcript.strip()) < 2:
                log.info("🔇 No meaningful speech recognized in this segment.")
                await ctx.send("Kheer: Aawaz saaf nahi aayi ya thoda dheeme bola tune, phir se bol na dhang se 🙄")
                return

            log.info(f"📝 User said: {transcript}")
            await ctx.send(f"🎤 *\"{transcript}\"*")

            if guild_id not in conversation_history:
                conversation_history[guild_id] = []

            conversation_history[guild_id].append(
                {
                    "role": "user",
                    "content": transcript,
                }
            )

            if len(conversation_history[guild_id]) > 20:
                conversation_history[guild_id] = conversation_history[guild_id][-20:]

            log.info("🧠 Generating savage response (Gemini)...")
            response_text = await llm.generate_response(conversation_history[guild_id])
            log.info(f"💬 Kheer: {response_text}")

            conversation_history[guild_id].append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

            await ctx.send(f"💬 **Kheer:** {response_text}")

            log.info("🔊 Synthesizing speech (ElevenLabs)...")
            audio_out = await tts.synthesize(response_text)

            if not audio_out:
                log.error("TTS returned no audio.")
                return

            log.info("▶️ Playing response in voice channel...")
            await play_audio(ctx, audio_out)

        except Exception as e:
            log.error(f"❌ Error handling conversation: {e}", exc_info=True)
        finally:
            sink.paused = False


# ─── Playback ───────────────────────────────────────────────────────────────

async def play_audio(ctx: commands.Context, audio_bytes: bytes):
    """Play ElevenLabs MP3 through the Discord voice connection."""
    vc = ctx.voice_client

    if not vc or not vc.is_connected():
        return

    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    )

    tmp_file.write(audio_bytes)
    tmp_file.close()

    try:
        source = discord.FFmpegPCMAudio(tmp_file.name)

        if vc.is_playing():
            vc.stop_playing()

        vc.play(source)

        while vc.is_playing():
            await asyncio.sleep(0.3)

    finally:
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


# ─── Reset ──────────────────────────────────────────────────────────────────

@bot.command(name="reset", help="Conversation history reset karo")
async def reset_history(ctx: commands.Context):
    guild_id = ctx.guild.id
    conversation_history[guild_id] = []

    await ctx.send(
        "Memory wipe kar diya! Ab mujhe kuch yaad nahi 🧹✨"
    )


# ─── Error Handler ──────────────────────────────────────────────────────────

@bot.event
async def on_command_error(ctx: commands.Context, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(
            "Ye kya command hai? Pehle seekh ke aa 🙄"
        )
    else:
        log.error(
            f"Command error: {error}",
            exc_info=True,
        )
        await ctx.send(
            f"Kuch gadbad ho gayi: `{error}` 😤"
        )


# ─── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        log.error(
            "❌ DISCORD_BOT_TOKEN not found in .env file!"
        )
        raise SystemExit(1)

    log.info("🚀 Starting Savage Girl Bot...")
    bot.run(token)