"""
AI Conversation Classifier - Lead Detection
Single file async implementation using Vertex AI Gemini REST API
"""

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# ============================================================================
# Configuration
# ============================================================================

class ClassificationType(str, Enum):
    LEAD = "lead"
    NOT_LEAD = "not_lead"
    NEEDS_INFO = "needs_info"


@dataclass
class ClassificationResult:
    classification: ClassificationType
    confidence: float
    reasoning: str
    is_lead: bool


@dataclass
class ConversationInput:
    """Input variables for classification prompt"""
    conversation_id: str
    messages: list[dict]  # [{"role": "customer"|"business", "content": "...", "timestamp": "..."}]
    source: str  # instagram, whatsapp, facebook, tiktok, email
    clinic_name: str  # Name of the clinic/practice
    clinic_type: str  # e.g., "dental clinic", "dermatology practice"
    services: list[str]  # List of services offered by the clinic


# ============================================================================
# Prompt Template
# ============================================================================

CLASSIFICATION_PROMPT = """
You are an AI assistant that classifies customer conversations for a {clinic_type} called "{clinic_name}".
Analyze the conversation and determine if this contact is a LEAD (potential customer).

## Clinic Services:
{formatted_services}

## Classification Rules:

**LEAD** - Mark as lead if the contact shows ANY of these signals:
- Asking about services, treatments, or procedures
- Inquiring about pricing or availability
- Wanting to book or schedule an appointment
- Asking about business hours or location
- Expressing a health concern or need related to clinic services
- Requesting a callback or more information
- Showing interest in specific offerings from the services list

**NOT_LEAD** - Mark as not lead if:
- Spam, promotional content, or advertisements
- Wrong number or misdirected messages
- Job seekers or salespeople
- Automated bot messages
- Clearly irrelevant to the clinic services

**NEEDS_INFO** - Mark as needs_info if:
- Message is too short or unclear to determine intent (e.g., just "hi" or "hello")
- Context is insufficient for classification
- Mixed signals that require clarification

## Conversation Context:

Source Platform: {source}

## Conversation Messages:

{formatted_messages}

## Response Format:

Respond with valid JSON only:
{{
    "classification": "lead" | "not_lead" | "needs_info",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this classification was chosen",
    "key_signals": ["list", "of", "key", "indicators"]
}}
"""


# ============================================================================
# Classifier
# ============================================================================

class LeadClassifier:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash"
    ):
        """
        Initialize the Lead Classifier with Vertex AI REST API.

        Args:
            api_key: Vertex AI API key
            model_name: Gemini model to use (default: gemini-2.0-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        # Vertex AI endpoint
        self.base_url = "https://aiplatform.googleapis.com/v1/publishers/google/models"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _format_messages(self, messages: list[dict]) -> str:
        """Format conversation messages for the prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            if timestamp:
                formatted.append(f"[{timestamp}] {role}: {content}")
            else:
                formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _format_services(self, services: list[str]) -> str:
        """Format services list for the prompt."""
        return "\n".join(f"- {service}" for service in services)

    def _build_prompt(self, conversation: ConversationInput) -> str:
        """Build the classification prompt with variables."""
        formatted_messages = self._format_messages(conversation.messages)
        formatted_services = self._format_services(conversation.services)

        return CLASSIFICATION_PROMPT.format(
            clinic_name=conversation.clinic_name,
            clinic_type=conversation.clinic_type,
            formatted_services=formatted_services,
            source=conversation.source,
            formatted_messages=formatted_messages
        )

    async def classify(self, conversation: ConversationInput) -> ClassificationResult:
        """
        Classify a conversation as lead or not.

        Args:
            conversation: ConversationInput with all conversation details

        Returns:
            ClassificationResult with classification, confidence, and reasoning
        """
        prompt = self._build_prompt(conversation)

        # Build API request for Vertex AI
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json"
            }
        }

        # Call Vertex AI API
        session = await self._get_session()
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            response.raise_for_status()
            response_data = await response.json()

        # Parse response
        text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)

        classification = ClassificationType(result["classification"])

        return ClassificationResult(
            classification=classification,
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            is_lead=classification == ClassificationType.LEAD
        )

    async def classify_batch(
        self, conversations: list[ConversationInput]
    ) -> list[ClassificationResult]:
        """
        Classify multiple conversations concurrently.

        Args:
            conversations: List of ConversationInput objects

        Returns:
            List of ClassificationResult objects
        """
        tasks = [self.classify(conv) for conv in conversations]
        return await asyncio.gather(*tasks)


# ============================================================================
# Usage Example
# ============================================================================

async def main():
    # Initialize classifier (uses GEMINI_API_KEY from .env)
    classifier = LeadClassifier()

    # Example clinic data
    dental_clinic = {
        "clinic_name": "Bright Smile Dental",
        "clinic_type": "dental clinic",
        "services": [
            "Teeth Whitening",
            "Dental Implants",
            "Root Canal Treatment",
            "Dental Crowns",
            "Invisalign",
            "Regular Checkups",
            "Emergency Dental Care"
        ]
    }

    medical_clinic = {
        "clinic_name": "City Health Medical Center",
        "clinic_type": "general medical practice",
        "services": [
            "General Consultation",
            "Annual Physical Exams",
            "Vaccinations",
            "Lab Tests",
            "Chronic Disease Management",
            "Minor Injury Treatment",
            "Referrals to Specialists"
        ]
    }

    # Example conversations to classify
    examples = [
        # Example 1: Clear lead - asking about specific service
        ConversationInput(
            conversation_id="conv_001",
            source="instagram",
            messages=[
                {"role": "customer", "content": "Hi, I saw your clinic on Instagram"},
                {"role": "business", "content": "Hello! How can we help you today?"},
                {"role": "customer", "content": "I'm interested in teeth whitening. How much does it cost and do you have availability next week?"},
            ],
            **dental_clinic
        ),

        # Example 2: Spam
        ConversationInput(
            conversation_id="conv_002",
            source="whatsapp",
            messages=[
                {"role": "customer", "content": "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize: bit.ly/scam123"},
            ],
            **medical_clinic
        ),

        # Example 3: Needs more info
        ConversationInput(
            conversation_id="conv_003",
            source="facebook",
            messages=[
                {"role": "customer", "content": "Hi"},
            ],
            **medical_clinic
        ),

        # Example 4: Lead with appointment request
        ConversationInput(
            conversation_id="conv_004",
            source="email",
            messages=[
                {"role": "customer", "content": "Hello, I've been experiencing back pain for the past week and would like to schedule an appointment with a specialist. What times do you have available this week?"},
            ],
            **dental_clinic
        ),
    ]

    try:
        # Classify all conversations concurrently
        print("=" * 60)
        print("LEAD CLASSIFICATION RESULTS (ASYNC)")
        print("=" * 60)

        results = await classifier.classify_batch(examples)

        for conv, result in zip(examples, results):
            print(f"\n--- Conversation: {conv.conversation_id} ({conv.source}) ---")
            print(f"Clinic: {conv.clinic_name}")
            print(f"Messages: {len(conv.messages)}")
            print(f"\nResult:")
            print(f"  Classification: {result.classification.value.upper()}")
            print(f"  Is Lead: {'YES' if result.is_lead else 'NO'}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Reasoning: {result.reasoning}")
            print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await classifier.close()


if __name__ == "__main__":
    asyncio.run(main())
