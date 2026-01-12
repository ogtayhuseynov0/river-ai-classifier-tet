"""Data models for synthetic data generation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import uuid
import json


@dataclass
class SyntheticService:
    """A service offered by a business."""
    name: str
    price: float
    currency: str = "USD"
    description: str = ""
    duration_minutes: int | None = None


@dataclass
class SyntheticBusiness:
    """A synthetic business with services."""
    business_id: str
    name: str
    business_type: str  # "law firm", "barbershop", "yacht charter", etc.
    services: list[SyntheticService]
    location: str = ""

    @classmethod
    def create(cls, name: str, business_type: str, services: list[dict], location: str = "") -> SyntheticBusiness:
        """Create from raw data."""
        return cls(
            business_id=f"bus_{uuid.uuid4().hex[:8]}",
            name=name,
            business_type=business_type,
            services=[SyntheticService(**s) for s in services],
            location=location
        )

    def get_service_names(self) -> list[str]:
        """Get list of service names for classifier."""
        return [s.name for s in self.services]

    def to_dict(self) -> dict:
        return {
            "business_id": self.business_id,
            "name": self.name,
            "business_type": self.business_type,
            "location": self.location,
            "services": [
                {
                    "name": s.name,
                    "price": s.price,
                    "currency": s.currency,
                    "description": s.description,
                    "duration_minutes": s.duration_minutes
                }
                for s in self.services
            ]
        }


@dataclass
class SyntheticContact:
    """A synthetic contact/lead."""
    contact_id: str
    first_name: str | None
    last_name: str | None
    persona: str  # serious_lead, curious_lead, price_shopper, spam, wrong_number, vague
    demographics: dict = field(default_factory=dict)  # city, country, language, occupation, etc.

    @classmethod
    def create(cls, first_name: str | None, last_name: str | None, persona: str, demographics: dict = None) -> SyntheticContact:
        """Create from raw data."""
        return cls(
            contact_id=f"con_{uuid.uuid4().hex[:8]}",
            first_name=first_name,
            last_name=last_name,
            persona=persona,
            demographics=demographics or {}
        )

    def get_display_name(self) -> str:
        """Get display name."""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else "Unknown"

    def to_dict(self) -> dict:
        return {
            "contact_id": self.contact_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "persona": self.persona,
            "demographics": self.demographics
        }


@dataclass
class SyntheticConversation:
    """A synthetic conversation with ground truth."""
    conversation_id: str
    business: SyntheticBusiness
    contact: SyntheticContact
    messages: list[dict]  # [{role: "customer"|"business", content: str, timestamp: str}]
    source: str  # instagram, whatsapp, facebook, email
    ground_truth: Literal["lead", "not_lead", "needs_info"]
    scenario_type: str  # direct_booking, price_inquiry, spam, etc.

    @classmethod
    def create(
        cls,
        business: SyntheticBusiness,
        contact: SyntheticContact,
        messages: list[dict],
        source: str,
        ground_truth: str,
        scenario_type: str
    ) -> SyntheticConversation:
        """Create from raw data."""
        return cls(
            conversation_id=f"conv_{uuid.uuid4().hex[:8]}",
            business=business,
            contact=contact,
            messages=messages,
            source=source,
            ground_truth=ground_truth,
            scenario_type=scenario_type
        )

    def to_conversation_input(self):
        """Convert to main.py ConversationInput format for classifier testing."""
        # Import here to avoid circular dependency
        try:
            from main import ConversationInput
            return ConversationInput(
                conversation_id=self.conversation_id,
                messages=self.messages,
                source=self.source,
                clinic_name=self.business.name,
                clinic_type=self.business.business_type,
                services=self.business.get_service_names()
            )
        except ImportError:
            # Return dict if ConversationInput not available
            return {
                "conversation_id": self.conversation_id,
                "messages": self.messages,
                "source": self.source,
                "clinic_name": self.business.name,
                "clinic_type": self.business.business_type,
                "services": self.business.get_service_names()
            }

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "business_id": self.business.business_id,
            "contact_id": self.contact.contact_id,
            "messages": self.messages,
            "source": self.source,
            "ground_truth": self.ground_truth,
            "scenario_type": self.scenario_type
        }


@dataclass
class SyntheticDataset:
    """A complete synthetic dataset."""
    business: SyntheticBusiness
    contacts: list[SyntheticContact]
    conversations: list[SyntheticConversation]
    metadata: dict = field(default_factory=dict)

    def to_json(self, filepath: str) -> None:
        """Export dataset to JSON file."""
        data = {
            "metadata": self.metadata,
            "business": self.business.to_dict(),
            "contacts": [c.to_dict() for c in self.contacts],
            "conversations": [c.to_dict() for c in self.conversations]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_csv(self, filepath: str) -> None:
        """Export conversations to CSV file."""
        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "conversation_id", "business_name", "business_type", "source",
                "contact_name", "persona", "ground_truth", "scenario_type",
                "message_count", "first_message", "last_message"
            ])

            # Rows
            for conv in self.conversations:
                first_msg = conv.messages[0]["content"][:100] if conv.messages else ""
                last_msg = conv.messages[-1]["content"][:100] if conv.messages else ""

                writer.writerow([
                    conv.conversation_id,
                    conv.business.name,
                    conv.business.business_type,
                    conv.source,
                    conv.contact.get_display_name(),
                    conv.contact.persona,
                    conv.ground_truth,
                    conv.scenario_type,
                    len(conv.messages),
                    first_msg,
                    last_msg
                ])

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        ground_truth_counts = {}
        channel_counts = {}
        persona_counts = {}

        for conv in self.conversations:
            ground_truth_counts[conv.ground_truth] = ground_truth_counts.get(conv.ground_truth, 0) + 1
            channel_counts[conv.source] = channel_counts.get(conv.source, 0) + 1
            persona_counts[conv.contact.persona] = persona_counts.get(conv.contact.persona, 0) + 1

        return {
            "total_conversations": len(self.conversations),
            "total_contacts": len(self.contacts),
            "business_name": self.business.name,
            "business_type": self.business.business_type,
            "services_count": len(self.business.services),
            "ground_truth_distribution": ground_truth_counts,
            "channel_distribution": channel_counts,
            "persona_distribution": persona_counts
        }
