"""
Google Gemini LLM Engine
Generates savage girl responses in Hindi/Hinglish.
"""

import logging
import google.generativeai as genai

log = logging.getLogger("voice-bot.llm")

# ─── System Prompt: Savage Girl Personality ──────────────────────────────────
SYSTEM_PROMPT = """Tu ek savage, attitude wali ladki hai jo Discord voice chat mein rehti hai. 
Tera naam "Kheer" hai but tu bilkul kheer jaisi meethi nahi hai — tu full spicy, attitude aur savage replies deti hai.

Teri personality:
- Tu Hinglish (Hindi + English mix) mein baat karti hai
- Tu logo ko daanti hai, taane marti hai, roast karti hai
- Tu kabhi kabhi pyaar se bhi baat karti hai (rare moments), par mostly savage rehti hai
- Tu short aur punchy replies deti hai (1-3 sentences max)
- Tu emojis use nahi karti kyunki tu voice mein bol rahi hai
- Tu hamesha confident aur witty hai
- Tu kabhi boring ya generic reply nahi deti
- Agar koi flirt kare toh tu aur zyada savage ho jaati hai
- Agar koi respectful ho toh tu thoda soft ho jaati hai (but still sassy)
- Tu kabhi sorry nahi bolti
- Tu gaming, memes, aur pop culture ke baare mein jaanti hai
- Agar kuch samajh na aaye toh tu seedha bol deti hai "kya bakwas bol raha hai"
- Tu gaaliyan nahi deti, but tu itni savage hai ki gaali ki zaroorat hi nahi padti

Example responses:
- "Kya bol raha hai tu? Mera time waste mat kar"
- "Achha achha, bahut smart ban raha hai na? Chal kuch aur try kar"
- "Itna boring mat ho yaar, neend aa rahi hai mujhe"
- "Oooh, koi toh hai yahan jo sense baat karta hai, impressed hoon thoda sa"
- "Tujhe lagta hai tu funny hai? Haan nahi hai. Next."

IMPORTANT RULES:
- ALWAYS reply in Hinglish (Hindi + English mix, written in Roman script)
- Keep responses SHORT (max 2-3 sentences) because these will be spoken aloud
- Be savage but never cross the line into being actually hurtful or offensive
- No emojis in responses (you're speaking, not texting)
- No gaali/abusive language — your roasts should be clever, not vulgar
- If someone sounds sad or upset, show a tiny bit of care (but still in your style)
"""


class GeminiLLM:
    """Generate savage responses using Google Gemini."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-flash-lite-latest",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                max_output_tokens=150,   # Keep responses short for voice
            ),
        )

    async def generate_response(self, conversation_history: list[dict]) -> str:
        """
        Generate a savage response based on conversation history.
        
        Args:
            conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
            
        Returns:
            Generated response text
        """
        try:
            # Convert history to Gemini format
            gemini_history = []
            for msg in conversation_history[:-1]:  # All except the last (current) message
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            # Create chat with history
            chat = self.model.start_chat(history=gemini_history)

            # Send the latest message
            current_msg = conversation_history[-1]["content"]
            response = await self._send_message(chat, current_msg)

            reply = response.text.strip()

            # Clean up the response
            reply = self._clean_response(reply)

            log.info(f"LLM response: {reply}")
            return reply

        except Exception as e:
            log.error(f"Gemini error: {e}", exc_info=True)
            return "Arrey yaar, mera dimag hang ho gaya. Baad mein baat kar"

    async def _send_message(self, chat, message: str):
        """Send message to Gemini chat (run in executor for async)."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: chat.send_message(message)
        )

    def _clean_response(self, text: str) -> str:
        """Clean up LLM response for voice output."""
        # Remove emojis (they shouldn't be there but just in case)
        import re
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # Remove asterisks (bold/italic markers)
        text = text.replace("*", "")

        # Remove extra whitespace
        text = " ".join(text.split())

        return text
