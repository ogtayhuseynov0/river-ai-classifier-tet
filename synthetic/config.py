"""Configuration for synthetic data generation."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""

    # Seed for reproducibility (None = random)
    seed: int | None = 42

    # Number of conversations to generate
    num_conversations: int = 100

    # Number of unique contacts to generate
    num_contacts: int = 50

    # Ground truth distribution (must sum to 1.0)
    lead_ratio: float = 0.4
    not_lead_ratio: float = 0.4
    needs_info_ratio: float = 0.2

    # Channel distribution (must sum to 1.0)
    channels: dict[str, float] = field(default_factory=lambda: {
        "instagram": 0.35,
        "whatsapp": 0.35,
        "facebook": 0.15,
        "email": 0.15
    })

    # Persona distribution for contacts (must sum to 1.0)
    personas: dict[str, float] = field(default_factory=lambda: {
        "serious_lead": 0.30,
        "curious_lead": 0.20,
        "price_shopper": 0.15,
        "spam": 0.15,
        "wrong_number": 0.10,
        "vague": 0.10
    })

    # Message count range per conversation
    min_messages: int = 2
    max_messages: int = 8

    # Gemini model to use for generation
    gemini_model: str = "gemini-2.0-flash-lite"

    def validate(self) -> None:
        """Validate configuration values."""
        # Check ratios sum to ~1.0
        ground_truth_sum = self.lead_ratio + self.not_lead_ratio + self.needs_info_ratio
        if abs(ground_truth_sum - 1.0) > 0.01:
            raise ValueError(f"Ground truth ratios must sum to 1.0, got {ground_truth_sum}")

        channel_sum = sum(self.channels.values())
        if abs(channel_sum - 1.0) > 0.01:
            raise ValueError(f"Channel ratios must sum to 1.0, got {channel_sum}")

        persona_sum = sum(self.personas.values())
        if abs(persona_sum - 1.0) > 0.01:
            raise ValueError(f"Persona ratios must sum to 1.0, got {persona_sum}")

        # Check counts
        if self.num_conversations < 1:
            raise ValueError("num_conversations must be at least 1")
        if self.num_contacts < 1:
            raise ValueError("num_contacts must be at least 1")
        if self.min_messages < 1:
            raise ValueError("min_messages must be at least 1")
        if self.max_messages < self.min_messages:
            raise ValueError("max_messages must be >= min_messages")

    def get_ground_truth_for_persona(self, persona: str) -> str:
        """Map persona to expected ground truth."""
        mapping = {
            "serious_lead": "lead",
            "curious_lead": "lead",
            "price_shopper": "lead",
            "spam": "not_lead",
            "wrong_number": "not_lead",
            "vague": "needs_info"
        }
        return mapping.get(persona, "needs_info")

    def get_scenario_for_persona(self, persona: str) -> str:
        """Get typical scenario type for persona."""
        mapping = {
            "serious_lead": "direct_booking",
            "curious_lead": "service_inquiry",
            "price_shopper": "price_inquiry",
            "spam": "spam_promotional",
            "wrong_number": "wrong_number",
            "vague": "vague_greeting"
        }
        return mapping.get(persona, "general_inquiry")
