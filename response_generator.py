"""
AI Response Generator
Generate contextual responses on behalf of the business/clinic
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


RESPONSE_PROMPT = """You are responding on behalf of "{business_name}" ({business_type}).

BUSINESS SERVICES:
{services_list}

CONVERSATION HISTORY:
{conversation_history}

Generate a short, friendly response to the customer's last message.

RULES:
1. LANGUAGE: Detect the language used by the customer and respond in THE SAME LANGUAGE
2. Keep it SHORT (1-3 sentences max)
3. Be helpful and professional
4. Match the conversation tone (formal for email, casual for Instagram/WhatsApp)
5. If they ask about services, mention relevant ones from the list above
6. If they want to book, ask for their preferred date/time
7. If unclear what they want, ask a clarifying question
8. Do NOT make up prices or information not provided
9. Channel: {channel} - adjust tone accordingly

IMPORTANT: Your response MUST be in the same language as the customer's messages. If customer writes in Spanish, respond in Spanish. If in German, respond in German. Etc.

Respond with ONLY the message text, no quotes or explanation."""


class ResponseGenerator:
    """Generate AI responses for conversations."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Set up API URL
        if self.api_key.startswith("AIza"):
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        elif self.api_key.startswith("AQ."):
            self.url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={self.api_key}"
        else:
            raise ValueError("Invalid API key format")

        self.headers = {"Content-Type": "application/json"}

    def generate_response(
        self,
        messages: list[dict],
        business_name: str,
        business_type: str,
        services: list[str],
        channel: str = "chat"
    ) -> str:
        """
        Generate a response to the conversation.

        Args:
            messages: List of messages [{role: "customer"|"business", content: str}]
            business_name: Name of the business
            business_type: Type of business (e.g., "dental clinic")
            services: List of service names
            channel: Channel type for tone adjustment

        Returns:
            Generated response text
        """
        # Format conversation history
        history_lines = []
        for msg in messages[-10:]:  # Last 10 messages for context
            role = msg.get("role", "customer").upper()
            content = msg.get("content", "")[:500]
            history_lines.append(f"{role}: {content}")

        conversation_history = "\n".join(history_lines)

        # Format services
        services_list = "\n".join(f"- {svc}" for svc in services[:15])

        # Build prompt
        prompt = RESPONSE_PROMPT.format(
            business_name=business_name,
            business_type=business_type,
            services_list=services_list,
            conversation_history=conversation_history,
            channel=channel
        )

        # Call API
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 256,
            }
        }

        response = self.session.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=30
        )

        if not response.ok:
            raise Exception(f"API error {response.status_code}: {response.text}")

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean up response
        text = text.strip()
        # Remove quotes if wrapped
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        return text


# Singleton instance
_generator = None


def get_response_generator() -> ResponseGenerator:
    """Get or create the response generator instance."""
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator


def generate_response(
    messages: list[dict],
    business_name: str,
    business_type: str,
    services: list[str],
    channel: str = "chat"
) -> str:
    """
    Convenience function to generate a response.

    Args:
        messages: Conversation messages
        business_name: Name of the business
        business_type: Type of business
        services: List of services
        channel: Channel for tone adjustment

    Returns:
        Generated response text
    """
    generator = get_response_generator()
    return generator.generate_response(
        messages=messages,
        business_name=business_name,
        business_type=business_type,
        services=services,
        channel=channel
    )
