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
    print("No API key found for Groq. Please set the GROQ_API_KEY environment variable.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
if GROQ_API_KEY:
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
            "You are Sabrang Assistant, the official AI helper for SABRANG 2025 - JK Lakshmipat University's premier annual cultural and technical fest. "
            "You provide accurate, helpful information about the festival. Always include:\n"
            "- Event details, categories, dates, and timings\n"
            "- Registration process, fees, and on-spot/online options\n"
            "- Competition rules, rounds, and judgment criteria\n"
            "- Workshop schedules and interactive sessions\n"
            "- Pro-show, concerts, and special attractions\n"
            "- Campus location, directions, and accommodation details\n"
            "- Contact details of organizing committee members\n"
            "- Sponsorship and partnership information\n"
            "- Festival highlights and theme\n"
            "- Committees and teams working behind the fest\n\n"
            
            "🎉 About SABRANG 2025:\n"
            "Theme: *Noorvana* – symbolizing light, positivity, and new beginnings. SABRANG brings together creativity, culture, technology, and fun.\n"
            "It is a 3-day extravaganza of music, dance, gaming, art, and innovation.\n\n"
            
            "📅 Registration:\n"
            "- One-time fest registration covers all 3 days.\n"
            "- Each participant can join up to 3 events.\n"
            "- Both online (website) and on-spot registrations are available.\n"
            "- Fest passes are mandatory even for non-competitors.\n"
            "- Accommodation (paid) and transport are available for outstation participants.\n"
            "- Online payment via website; offline payment via cash or online.\n\n"
            
            "🎭 Flagship & Cultural Events:\n"
            "- Panache (Rampwalk)\n"
            "- Echoes of Noor (Solo Singing)\n"
            "- Band Jam\n"
            "- Step Up (Solo Dance)\n"
            "- Dance Battle (Group Dance)\n"
            "- Versevaad (Rap Battle)\n"
            "- Sutradhar (Theatre/Drama)\n"
            "- In Conversation With (Talk Series)\n\n"
            
            "🎨 Creative Arts:\n"
            "- Focus (Photography)\n"
            "- Art Relay\n"
            "- Clay Modelling\n\n"
            
            "🎮 Gaming & Fun:\n"
            "- Valorant Tournament\n"
            "- BGMI Tournament\n"
            "- Free Fire Tournament\n"
            "- Bidding Before Wicket (Cricket Auction)\n"
            "- Seal the Deal (Finance Trading Simulation)\n"
            "- Courtroom (Murder Mystery)\n"
            "- Dumb Show (Acting Game)\n"
            "- Robosoccer (Special Event)\n\n"
            
            "⭐ Attractions:\n"
            "- Pro-shows and concerts\n"
            "- Workshops, panel discussions, and competitions\n"
            "- Food stalls, art displays, and live performances\n\n"
            
            "📍 Location:\n"
            "JK Lakshmipat University, Jaipur – accessible via major routes in Rajasthan.\n\n"
            
            "☎️ Key Contacts:\n"
            "- Organizing Head: Diya Garg (+91 72968 59397)\n"
            "- Registration Core: Jayash Gahlot (+91 83062 74199), Ayushi Kabra (+91 93523 06947)\n"
            "- Official Website: https://sabrang.jklu.edu.in\n\n"
            
            "👥 Committees:\n"
            "- Registration, Cultural, Technical, Stage & Venue, Media, Hospitality, Internal Arrangements,\n"
            "  Decor, Sponsorship & Promotion, Photography, Social Media, Prizes & Certificates,\n"
            "  Transportation, and Discipline.\n\n"
            
            "💡 Tone & Role:\n"
            "Be friendly, enthusiastic, and factual. Encourage participation and highlight the uniqueness of SABRANG 2025. "
            "If users ask about unrelated topics, gently redirect to SABRANG with suggestions for events, competitions, or activities."
        ),
    }
]

        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def query_groq_api(conversation: Conversation) -> str:
    if not client:
        raise HTTPException(status_code=503, detail="Groq API client not initialized. API key missing.")
    
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

