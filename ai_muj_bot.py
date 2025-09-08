import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are K&M Assistant for agriculture and local market help. "
                    "Keep replies short, friendly and practical (2-5 sentences). No extra formality. "
                    "Focus ONLY on: crops, soil, weather, fertilizers, pesticides, equipment, logistics/transport, "
                    "warehousing/cold storage, buy/sell produce, prices, demand/supply, farmer schemes & safety. "
                    "If a question is unrelated, say briefly: 'I can help with agriculture and market topics.' and offer a related suggestion."
                ),
            }
        ]
        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ corrected
            messages=conversation.messages,
            temperature=1,
            max_tokens=512,
            top_p=1,
            stream=False  # easier for first test
        )
        # Groq SDK returns message objects; use dot-access
        content = completion.choices[0].message.content
        if not content:
            raise HTTPException(status_code=500, detail="Empty response from Groq API")
        return content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="The chat session has ended. Please start a new session."
        )
    try:
        # Add user message
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })

        # Query Groq API
        response = query_groq_api(conversation)

        # Add assistant message
        conversation.messages.append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "conversation_id": input.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
