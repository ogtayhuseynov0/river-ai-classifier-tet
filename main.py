"""
AI Conversation Classifier - Lead Detection
Using Google AI Studio or Vertex AI
OPTIMIZED VERSION - Fixed 15 second delay issues
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ClassificationResult:
    classification: str
    confidence: float
    reasoning: str
    key_signals: list[str]
    is_lead: bool


@dataclass
class ExtractedData:
    """Extracted customer/lead data from conversation"""
    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    date_of_birth: str | None = None  # YYYY-MM-DD format
    gender: str | None = None  # male, female, other
    gender_identity: str | None = None
    street: str | None = None
    house_number: str | None = None
    post_code: str | None = None
    city: str | None = None
    country: str | None = None
    language: str | None = None  # ISO 639-1: en, fr, es, de, etc.
    occupation: str | None = None
    timezone: str | None = None  # e.g., Europe/London, America/New_York
    locale: str | None = None  # e.g., en_US, fr_FR
    metadata: dict | None = None  # Free-form key-value pairs


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
# Prompt Template (Optimized - removed extra whitespace)
# ============================================================================

CLASSIFICATION_PROMPT = """Classify this conversation for {clinic_type} "{clinic_name}".

ONLY these services are offered:
{formatted_services}

LEAD - asking about OUR services listed above, pricing, appointments, or related health concerns
NOT_LEAD - spam, wrong number, job seekers, OR asking about services WE DON'T OFFER
NEEDS_INFO - too short/unclear (just "hi")

Confidence guide:
- 0.95+ = explicit intent (e.g. "I want to book an appointment for X")
- 0.80-0.94 = strong signals (asking prices, availability)
- 0.60-0.79 = moderate signals (general interest, unclear which service)
- 0.40-0.59 = weak signals (vague message, mixed intent)
- <0.40 = very uncertain

Conversation ({source}):
{formatted_messages}

JSON: {{"classification":"lead|not_lead|needs_info","confidence":0.0-1.0,"reasoning":"brief","key_signals":["signal"]}}"""


EXTRACTION_PROMPT = """Extract customer/lead information from this conversation. Only extract what is explicitly mentioned or clearly implied.

Conversation:
{formatted_messages}

Extract these fields (use null if not found):
- first_name, last_name, middle_name: Person's name
- date_of_birth: Format YYYY-MM-DD
- gender: male, female, or other
- gender_identity: If specified differently
- street, house_number, post_code, city, country: Address components
- language: ISO 639-1 code (en, fr, es, de, etc.) - detect from conversation language
- occupation: Job or profession
- timezone: e.g., Europe/London, America/New_York
- locale: e.g., en_US, fr_FR
- metadata: Object with any other relevant info (phone, email, service_interested, appointment_preference, etc.)

JSON: {{"first_name":null,"last_name":null,"middle_name":null,"date_of_birth":null,"gender":null,"gender_identity":null,"street":null,"house_number":null,"post_code":null,"city":null,"country":null,"language":null,"occupation":null,"timezone":null,"locale":null,"metadata":null}}"""


# ============================================================================
# Classifier (Optimized)
# ============================================================================

class LeadClassifier:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash-lite",
        project_id: str = None,
        location: str = "us-central1"
    ):
        """
        Initialize the Lead Classifier.

        Supports two modes:
        - Google AI Studio: API key starting with 'AIza...'
        - Vertex AI: Uses Application Default Credentials (ADC)

        Args:
            api_key: Google AI Studio API key (optional, starts with AIza)
            model_name: Gemini model to use (default: gemini-2.0-flash-lite)
            project_id: GCP project ID (for Vertex AI mode)
            location: GCP region (default: us-central1)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location

        # ✅ FIX 1: Reuse HTTP session (connection pooling)
        self.session = requests.Session()
        
        # ✅ FIX 2: Cache for credentials (Vertex AI mode)
        self._credentials = None
        self._auth_req = None

        # Detect mode based on API key format
        if self.api_key and self.api_key.startswith("AIza"):
            # Google AI Studio mode (API key in URL)
            self.mode = "ai_studio"
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            self.url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            self.headers = {"Content-Type": "application/json"}
        elif self.api_key and self.api_key.startswith("AQ."):
            # Vertex AI with API key mode
            self.mode = "vertex_ai_key"
            self.base_url = "https://aiplatform.googleapis.com/v1/publishers/google/models"
            self.url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            self.headers = {"Content-Type": "application/json"}
        else:
            # Vertex AI mode (uses ADC/OAuth2)
            self.mode = "vertex_ai"
            self.base_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{location}/publishers/google/models"
            self.url = f"{self.base_url}/{self.model_name}:generateContent"
            # ✅ FIX 3: Pre-load credentials at init (not on every request)
            self._init_credentials()

        print(f"[LeadClassifier] Initialized in {self.mode} mode with model {self.model_name}")

    def _init_credentials(self):
        """Initialize credentials once at startup (Vertex AI only)."""
        try:
            import google.auth
            import google.auth.transport.requests
            
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._auth_req = google.auth.transport.requests.Request()
            # Initial token refresh
            self._credentials.refresh(self._auth_req)
            print("[LeadClassifier] Credentials initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to initialize credentials: {e}")

    def _get_headers(self) -> dict:
        """Get headers, refreshing token only if needed (Vertex AI)."""
        if self.mode == "vertex_ai":
            # ✅ FIX 4: Only refresh token if expired
            if not self._credentials.valid:
                self._credentials.refresh(self._auth_req)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._credentials.token}"
            }
        return self.headers

    def _format_messages(self, messages: list[dict]) -> str:
        """Format conversation messages for the prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Sanitize content to prevent JSON issues
            content = content.replace("\n", " ").replace("\r", " ").replace('"', "'")
            content = content[:500]  # Limit message length
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

    def classify(self, conversation: ConversationInput) -> ClassificationResult:
        """
        Classify a conversation as lead or not.

        Args:
            conversation: ConversationInput with all conversation details

        Returns:
            ClassificationResult with classification, confidence, and reasoning
        """
        prompt = self._build_prompt(conversation)

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": 256,  # ✅ FIX 5: Reduced from 1024 (you only need ~100 tokens)
                "responseMimeType": "application/json",
            }
        }

        # ✅ FIX 6: Use session for connection reuse
        response = self.session.post(
            self.url,
            json=payload,
            headers=self._get_headers(),
            timeout=30  # Add timeout to prevent hanging
        )

        if not response.ok:
            raise Exception(f"{response.status_code}: {response.text}")

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Parse response with robust error handling
        result = self._parse_json_response(text)

        # Normalize classification to lowercase
        classification = str(result.get("classification", "needs_info")).lower()
        # Handle variations
        if classification in ["lead", "is_lead", "yes"]:
            classification = "lead"
        elif classification in ["not_lead", "notlead", "no", "spam"]:
            classification = "not_lead"
        else:
            classification = "needs_info"

        return ClassificationResult(
            classification=classification,
            confidence=float(result.get("confidence", 0.5)),
            reasoning=str(result.get("reasoning", "")),
            key_signals=result.get("key_signals", []) or [],
            is_lead=classification == "lead"
        )

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON response with multiple fallback strategies."""
        import re

        # Strategy 1: Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result[0] if result else {}
            return result
        except:
            pass

        # Strategy 2: Find JSON object with brace matching
        try:
            start = text.find('{')
            if start != -1:
                depth = 0
                for i, char in enumerate(text[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            result = json.loads(text[start:i+1])
                            if isinstance(result, list):
                                return result[0] if result else {}
                            return result
        except:
            pass

        # Strategy 3: Extract key fields with regex
        try:
            classification = "needs_info"
            confidence = 0.5
            reasoning = ""

            # Find classification
            class_match = re.search(r'"classification"\s*:\s*"([^"]+)"', text)
            if class_match:
                classification = class_match.group(1).lower()

            # Find confidence
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            if conf_match:
                confidence = float(conf_match.group(1))

            # Find reasoning
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            if reason_match:
                reasoning = reason_match.group(1)

            return {
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "key_signals": []
            }
        except:
            pass

        # Fallback
        return {
            "classification": "needs_info",
            "confidence": 0.5,
            "reasoning": "Could not parse AI response",
            "key_signals": []
        }

    def extract(self, conversation: ConversationInput) -> ExtractedData:
        """
        Extract customer/lead data from a conversation.

        Args:
            conversation: ConversationInput with conversation details

        Returns:
            ExtractedData with extracted fields
        """
        formatted_messages = self._format_messages(conversation.messages)
        prompt = EXTRACTION_PROMPT.format(formatted_messages=formatted_messages)

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
            }
        }

        response = self.session.post(
            self.url,
            json=payload,
            headers=self._get_headers(),
            timeout=30
        )

        if not response.ok:
            raise Exception(f"{response.status_code}: {response.text}")

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Parse response
        result = self._parse_json_response(text)

        return ExtractedData(
            first_name=result.get("first_name"),
            last_name=result.get("last_name"),
            middle_name=result.get("middle_name"),
            date_of_birth=result.get("date_of_birth"),
            gender=result.get("gender"),
            gender_identity=result.get("gender_identity"),
            street=result.get("street"),
            house_number=result.get("house_number"),
            post_code=result.get("post_code"),
            city=result.get("city"),
            country=result.get("country"),
            language=result.get("language"),
            occupation=result.get("occupation"),
            timezone=result.get("timezone"),
            locale=result.get("locale"),
            metadata=result.get("metadata")
        )

    def classify_and_extract(self, conversation: ConversationInput) -> tuple[ClassificationResult, ExtractedData]:
        """
        Classify and extract data from a conversation in one call.

        Returns:
            Tuple of (ClassificationResult, ExtractedData)
        """
        classification = self.classify(conversation)
        extracted = self.extract(conversation)
        return classification, extracted

    def classify_batch(
        self, conversations: list[ConversationInput]
    ) -> list[ClassificationResult]:
        """
        Classify multiple conversations.

        Args:
            conversations: List of ConversationInput objects

        Returns:
            List of ClassificationResult objects
        """
        return [self.classify(conv) for conv in conversations]


# ============================================================================
# Usage Example
# ============================================================================

def main():
    # Initialize classifier ONCE (uses GEMINI_API_KEY from .env)
    # ✅ Important: Create classifier once, reuse for all requests
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

    # Classify all conversations with timing
    print("=" * 60)
    print("LEAD CLASSIFICATION RESULTS (OPTIMIZED)")
    print("=" * 60)

    try:
        total_start = time.time()

        for conv in examples:
            start_time = time.time()
            result = classifier.classify(conv)
            elapsed = time.time() - start_time

            print(f"\n--- Conversation: {conv.conversation_id} ({conv.source}) ---")
            print(f"Clinic: {conv.clinic_name}")
            print(f"Messages: {len(conv.messages)}")
            print(f"\nResult:")
            print(f"  Classification: {result.classification.upper()}")
            print(f"  Is Lead: {'YES ✓' if result.is_lead else 'NO ✗'}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Key Signals: {', '.join(result.key_signals)}")
            print(f"  ⏱️  Response Time: {elapsed:.2f}s")  # ✅ Now should be 0.5-2s
            print("-" * 60)

        total_elapsed = time.time() - total_start
        print(f"\n✅ Total time for {len(examples)} classifications: {total_elapsed:.2f}s")
        print(f"✅ Average per classification: {total_elapsed/len(examples):.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
