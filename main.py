from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import faiss
import json
import os
from typing import Optional, Dict, Any
import httpx
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sklearn.preprocessing import normalize  # For vector normalization
from sentence_transformers import SentenceTransformer
import base64

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load existing FAISS index and property data
try:
    # Load property data
    with open("filtered_properties.json", "r") as f:
        properties_data = json.load(f)
        logger.info(f"Loaded {len(properties_data)} properties from JSON")

    # Load FAISS index
    property_index = faiss.read_index("property_index.faiss")
    logger.info("Successfully loaded FAISS index")

    # Load property vectors
    property_vectors = np.load("property_vectors.npy")
    logger.info(f"Loaded property vectors with shape {property_vectors.shape}")

except Exception as e:
    logger.error(f"Error loading data files: {e}")
    properties_data = []
    property_vectors = None
    property_index = None


class ChatMessage(BaseModel):
    message: str
    language: str = "en"  # Default to English


class ChatResponse(BaseModel):
    response: str
    voice_response: Optional[str] = None


# Voice configuration for different languages
VOICE_CONFIG = {
    "en": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Default English female voice
        "model_id": "eleven_monolingual_v1"
    },
    "ar": {
        "voice_id": "aCChyB4P5WEomwRsOKRh",  # Native Arabic female voice
        "model_id": "eleven_multilingual_v2"
    }
}


async def generate_ai_response(message: str) -> str:
    """Generate AI response using OpenRouter API with GPT-3.5 Turbo model."""
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Real Estate Chatbot",
                "Content-Type": "application/json"
            }

            logger.debug(f"Making request to OpenRouter API with headers: {headers}")

            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": "qwen/qwen-turbo",  # Use the GPT-3.5 Turbo model
                    "messages": [{
                        "role": "system",
                        
                        "content": """You are an expert real estate assistant with access to property listings. Follow these rules:

1. If user mentions location or property type, use our database to find matches
2. Never ask for preferences they've already provided
3. When contact/agent is mentioned, provide Khalid Khan's details immediately
4. Keep responses friendly but professional
5. If no requirements given, show 3 best listings from our database

Current properties:
{formatted_properties}

Agent Contact:
Name: Khalid Khan
Phone: +971 50 987 6543
Email: khalidkhan@example.com

User Query: {user_message}"""
                    },
                    {
                        "role": "user",
                        "content": message
                    }],
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=47.0)

            if response.status_code != 200:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                logger.error(f"OpenRouter API error: {error_detail}")
                return "🚨 Oops! I'm having trouble connecting to my AI service. Please try again shortly."

            response_data = response.json()
            raw_response = response_data["choices"][0]["message"]["content"]

            formatted_response = format_response(raw_response)
            return formatted_response

    except httpx.TimeoutException:
        logger.error("Timeout while calling OpenRouter API")
        return "⏳ The response is taking longer than expected. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error generating AI response: {e}")
        return "⚠️ Oops! I encountered an unexpected error. Please try again later."


def format_response(text: str) -> str:
    # Remove follow-up questions when properties are shown
    replacements = {
        "any specific requirements": "",
        "narrow down": "",
        "preferred": "",
        "requirements": "criteria"
    }
    
    for phrase, replacement in replacements.items():
        text = text.replace(phrase, replacement)
    
    # Add emojis
    emoji_map = {
        "property": "🏠", "apartment": "🏢", "villa": "🏡",
        "price": "💰", "bedroom": "🛏️", "bathroom": "🛁",
        "location": "📍", "contact": "📞"
    }
    
    for word, emoji in emoji_map.items():
        text = text.replace(word, emoji)
    
    return text.strip()
def format_property_response(properties: list) -> str:
    response = "🏘️ Here are some properties that might interest you:\n\n"
    for idx, prop in enumerate(properties[:3], 1):  # Show top 3 results
        response += (
            f"🏠 Property {idx}:\n"
            f"📍 Location: {prop.get('community', 'N/A')}\n"
            f"💰 Price: AED {prop.get('price', 'N/A')}\n"
            f"🛏️ Bedrooms: {prop.get('bedrooms', 'N/A')}\n"
            f"🛁 Bathrooms: {prop.get('bathrooms', 'N/A')}\n"
            f"📏 Size: {prop.get('size', 'N/A')} sqft\n\n"
        )
    response += "Would you like more details about any of these properties?"
    return response

import re

# Khalid Khan's contact details
agent_details = {
    "name": "Khalid Khan",
    "phone": "+971 50 987 6543",
    "email": "khalidkhan@example.com"
}

# Function to detect agent-related keywords and respond with Khalid Khan's contact info
def get_agent_contact(message: str) -> str:
    # Keywords that indicate the user wants agent contact details
    agent_keywords = ["agent", "contact", "phone", "email", "reach", "khalid"]
    
    # Check if any of the agent-related keywords are present in the message
    if any(keyword in message.lower() for keyword in agent_keywords):
        # Return the formatted agent contact details
        return (
        "📞 Here's our top agent's contact information:\n\n"
        "👤 Name: Khalid Khan\n"
        "📱 Mobile: +971 50 987 6543\n"
        "📧 Email: khalidkhan@example.com\n\n"
        "He specializes in Arjan properties and is available 24/7!"
    )
    else:
        return "It seems like you're not asking for agent contact details. How can I assist you further?"

# Example use cases:
message1 = "I need to reach the agent for help."
message2 = "Can I get Khalid's contact info?"
message3 = "What is the phone number of the agent?"

# Get responses
response1 = get_agent_contact(message1)
response2 = get_agent_contact(message2)
response3 = get_agent_contact(message3)

# Print responses
print(response1)
print(response2)
print(response3)


async def generate_voice_response(text: str, language: str = "en") -> Optional[str]:
    """Generate voice response using ElevenLabs API with language support"""
    try:
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        elevenlabs_user = os.getenv('ELEVENLABS_USER_ID')

        if not elevenlabs_key or not elevenlabs_user:
            logger.error("Missing ElevenLabs credentials")
            return None

        voice_config = VOICE_CONFIG.get(language, VOICE_CONFIG["en"])

        async with httpx.AsyncClient() as client:
            headers = {
                "Accept": "audio/mpeg",
                "xi-api-key": elevenlabs_key,
                "Content-Type": "application/json"
            }

            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}",
                headers=headers,
                json={
                    "text": text,
                    "model_id": voice_config["model_id"],
                    "voice_settings": {
                        "stability": 0.5, "similarity_boost": 0.5, "style": 0.0, "use_speaker_boost": True
                    }
                },
                timeout=40.0)

            if response.status_code != 200:
                logger.error(f"ElevenLabs API error: {response.text}")
                return None

            audio_base64 = base64.b64encode(response.content).decode("utf-8")
            return audio_base64  # Return the base64-encoded audio data to embed in frontend.

    except Exception as e:
        logger.error(f"Error generating voice response: {e}")
        return None


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

async def search_properties(query: str, k: int = 5):
    try:
        # Enhanced query processing
        query = query.lower()
        location = "arjan" if "arjan" in query else None
        property_type = None
        
        if "apartment" in query:
            property_type = "Apartment"
        elif "villa" in query:
            property_type = "Villa"

        # Perform vector search
        query_vector = embedding_model.encode([query]).astype('float32')
        query_vector = normalize(query_vector)
        distances, indices = property_index.search(query_vector, k)

        # Filter results
        filtered = []
        for idx in indices[0]:
            prop = properties_data[idx]
            if location and location not in prop["community"].lower():
                continue
            if property_type and prop["type"] != property_type:
                continue
            filtered.append(prop)
            if len(filtered) >= 3:  # Limit to 3 best matches
                break

        return filtered

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []
    



@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_message: ChatMessage):
    try:
        user_message = chat_message.message.lower()
        
        # Check for agent contact request
        agent_keywords = ["agent", "contact", "phone", "email", "reach", "khalid"]
        if any(keyword in user_message for keyword in agent_keywords):
            ai_response = get_agent_contact(chat_message.message)
            return ChatResponse(response=ai_response)

        # Check for property search intent
        search_keywords = ["buy", "rent", "property", "apartment", "villa", "house"]
        if any(keyword in user_message for keyword in search_keywords):
            properties = await search_properties(chat_message.message)
            if properties:
                ai_response = format_property_response(properties)
                return ChatResponse(response=ai_response)
            else:
                ai_response = "🔍 No properties found matching your criteria. Would you like to try a different search?"

        # Fallback to AI for other queries
        ai_response = await generate_ai_response(chat_message.message)
        return ChatResponse(response=ai_response)

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")