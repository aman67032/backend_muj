import os
from typing import List, Dict, Literal, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()

# -------- Config --------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

# Comma-separated origins env support. Fallback to specific prod + localhost.
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_ENV:
    ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "https://sabrang.jklu.edu.in",
        "http://localhost:3000",
        "https://localhost:3000",
    ]

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# -------- App --------
app = FastAPI(title="Sabrang Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,     # be explicit in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Groq Client --------
client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY not set. /chat/ will return 503 until configured.")

# -------- Data Models --------
Role = Literal["user", "assistant", "system"]

class UserInput(BaseModel):
    message: str = Field(..., min_length=1)
    role: Role = "user"
    conversation_id: str = Field(..., min_length=3)

class Conversation:
    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Sabrang Assistant, the official AI helper for SABRANG 2025 - JK Lakshmipat "
                    "University's premier annual cultural and technical fest. Provide accurate, helpful, "
                    "concise information. You can cover: events, categories, dates, timings; registration "
                    "process/fees; rules/rounds/judgement criteria; workshops; pro-shows; campus/directions/"
                    "accommodation; contacts; sponsorship; highlights/theme; committees.\n\n"
                    "Theme: Noorvana – light, positivity, new beginnings. 3-day extravaganza of music, "
                    "dance, gaming, art, innovation.\n\n"
                    "Key contacts:\n"
                    "- Organizing Head: Diya Garg (+91 72968 59397)\n"
                    "- Registration Core: Jayash Gahlot (+91 83062 74199), Ayushi Kabra (+91 93523 06947)\n"
                    "- Official Website: https://sabrang.jklu.edu.in\n\n"
                    "Be friendly, enthusiastic, factual. Redirect unrelated queries back to Sabrang."
                ),
            }
        ]
        self.active: bool = True
        self.max_history: int = 30  # cap growth

    def add(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Keep last N messages + system
        if len(self.messages) > self.max_history + 1:
            # preserve first system message
            system = self.messages[0]
            self.messages = [system] + self.messages[-self.max_history:]

conversations: Dict[str, Conversation] = {}

def get_or_create_conversation(conversation_id: str) -> Conversation:
    convo = conversations.get(conversation_id)
    if not convo:
        convo = Conversation()
        conversations[conversation_id] = convo
    return convo

# -------- Groq Call --------
def query_groq_api(conversation: Conversation) -> str:
    if client is None:
        raise HTTPException(status_code=503, detail="Groq API client not initialized (missing GROQ_API_KEY).")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation.messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=1,
            stream=False,
        )
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise HTTPException(status_code=500, detail="Empty response from Groq API.")
        return content.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {e}")

# -------- Routes --------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "has_api_key": bool(GROQ_API_KEY),
        "allowed_origins": ALLOWED_ORIGINS,
    }

@app.post("/chat/")
def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)
    if not conversation.active:
        raise HTTPException(status_code=400, detail="Chat session ended. Please start a new session.")

    # Add user message
    conversation.add(role=input.role, content=input.message)

    # Get assistant reply
    reply = query_groq_api(conversation)

    # Add assistant message to history
    conversation.add(role="assistant", content=reply)

    return {
        "response": reply,
        "conversation_id": input.conversation_id,
    }

# -------- Local Dev Entry (optional) --------
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 for Railway/containers; adjust port as needed
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)