"""Main synthetic data generator using Gemini AI."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from .config import GenerationConfig
from .models import (
    SyntheticBusiness,
    SyntheticContact,
    SyntheticConversation,
    SyntheticDataset,
)
from .prompts import get_business_prompt, get_contacts_prompt, get_conversation_prompt

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class SyntheticDataGenerator:
    """Generate synthetic business data and conversations using Gemini AI."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """
        Initialize the generator.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Set up API URL based on key type
        if self.api_key.startswith("AIza"):
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        elif self.api_key.startswith("AQ."):
            self.url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={self.api_key}"
        else:
            raise ValueError("Invalid API key format. Must start with 'AIza' or 'AQ.'")

        self.headers = {"Content-Type": "application/json"}

    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        progress_callback=None
    ) -> SyntheticDataset:
        """
        Generate a complete synthetic dataset from a user prompt.

        Args:
            prompt: Natural language description of the business
            config: Generation configuration (optional)
            progress_callback: Optional callback(step, total, message) for progress

        Returns:
            SyntheticDataset with business, contacts, and conversations
        """
        config = config or GenerationConfig()
        config.validate()

        # Initialize random with seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)

        total_steps = 3 + config.num_conversations
        current_step = 0

        def report_progress(message: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, message)
            print(f"[{current_step}/{total_steps}] {message}")

        # Step 1: Generate business
        report_progress("Generating business...")
        business = self._generate_business(prompt)

        # Step 2: Generate contacts
        report_progress("Generating contacts...")
        contacts = self._generate_contacts(config.num_contacts, business.business_type, config.personas)

        # Step 3: Generate conversations
        conversations = []
        for i in range(config.num_conversations):
            report_progress(f"Generating conversation {i + 1}/{config.num_conversations}...")

            # Select contact, channel, and determine ground truth
            contact = random.choice(contacts)
            channel = self._weighted_choice(config.channels)
            ground_truth = config.get_ground_truth_for_persona(contact.persona)
            scenario = config.get_scenario_for_persona(contact.persona)
            num_messages = random.randint(config.min_messages, config.max_messages)

            try:
                conv = self._generate_conversation(
                    business=business,
                    contact=contact,
                    channel=channel,
                    ground_truth=ground_truth,
                    scenario_type=scenario,
                    num_messages=num_messages
                )
                conversations.append(conv)
            except Exception as e:
                print(f"  Warning: Failed to generate conversation: {e}")
                continue

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Create dataset
        dataset = SyntheticDataset(
            business=business,
            contacts=contacts,
            conversations=conversations,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "prompt": prompt,
                "config": {
                    "seed": config.seed,
                    "num_conversations": config.num_conversations,
                    "num_contacts": config.num_contacts,
                    "lead_ratio": config.lead_ratio,
                    "not_lead_ratio": config.not_lead_ratio,
                    "needs_info_ratio": config.needs_info_ratio,
                }
            }
        )

        print(f"\nGeneration complete! {len(conversations)} conversations created.")
        return dataset

    def _generate_business(self, user_prompt: str) -> SyntheticBusiness:
        """Generate business from user prompt."""
        prompt = get_business_prompt(user_prompt)
        response = self._call_gemini(prompt)
        data = self._parse_json(response)

        return SyntheticBusiness.create(
            name=data.get("name", "Generated Business"),
            business_type=data.get("business_type", "business"),
            services=data.get("services", []),
            location=data.get("location", "")
        )

    def _generate_contacts(
        self,
        num_contacts: int,
        business_type: str,
        personas: dict[str, float]
    ) -> list[SyntheticContact]:
        """Generate contacts with various personas."""
        prompt = get_contacts_prompt(num_contacts, business_type, personas)
        response = self._call_gemini(prompt, max_tokens=2048)
        data = self._parse_json(response)

        if not isinstance(data, list):
            data = [data]

        contacts = []
        for item in data:
            contacts.append(SyntheticContact.create(
                first_name=item.get("first_name"),
                last_name=item.get("last_name"),
                persona=item.get("persona", "curious_lead"),
                demographics=item.get("demographics", {})
            ))

        return contacts

    def _generate_conversation(
        self,
        business: SyntheticBusiness,
        contact: SyntheticContact,
        channel: str,
        ground_truth: str,
        scenario_type: str,
        num_messages: int
    ) -> SyntheticConversation:
        """Generate a single conversation."""
        services_list = ", ".join(s.name for s in business.services[:5])

        prompt = get_conversation_prompt(
            channel=channel,
            business_name=business.name,
            business_type=business.business_type,
            services_list=services_list,
            contact_name=contact.get_display_name(),
            persona=contact.persona,
            ground_truth=ground_truth,
            scenario_type=scenario_type,
            num_messages=num_messages
        )

        response = self._call_gemini(prompt)
        messages = self._parse_json(response)

        if not isinstance(messages, list):
            messages = [messages]

        # Ensure proper message format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "customer"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat())
            })

        return SyntheticConversation.create(
            business=business,
            contact=contact,
            messages=formatted_messages,
            source=channel,
            ground_truth=ground_truth,
            scenario_type=scenario_type
        )

    def _call_gemini(self, prompt: str, max_tokens: int = 1024) -> str:
        """Make a call to Gemini API."""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,  # Higher for more creative generation
                "topP": 0.9,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            }
        }

        response = self.session.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=60
        )

        if not response.ok:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_json(self, text: str) -> dict | list:
        """Parse JSON response with fallback strategies."""
        import re

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
            r'\[[\s\S]*\]'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Try to fix common issues
        text = text.strip()
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from response: {text[:200]}...")

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """Select item based on weights."""
        items = list(weights.keys())
        probs = list(weights.values())
        return random.choices(items, weights=probs, k=1)[0]


# CLI entry point
def main():
    """Command line interface for synthetic data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic business data")
    parser.add_argument("prompt", help="Business description prompt")
    parser.add_argument("--num-conversations", "-n", type=int, default=10, help="Number of conversations")
    parser.add_argument("--num-contacts", "-c", type=int, default=20, help="Number of contacts")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", default="synthetic/output", help="Output directory")
    parser.add_argument("--format", "-f", nargs="+", default=["json"], choices=["json", "csv"], help="Output formats")

    args = parser.parse_args()

    config = GenerationConfig(
        seed=args.seed,
        num_conversations=args.num_conversations,
        num_contacts=args.num_contacts
    )

    generator = SyntheticDataGenerator()
    dataset = generator.generate(args.prompt, config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "json" in args.format:
        filepath = output_dir / f"dataset_{timestamp}.json"
        dataset.to_json(str(filepath))
        print(f"Saved JSON to: {filepath}")

    if "csv" in args.format:
        filepath = output_dir / f"dataset_{timestamp}.csv"
        dataset.to_csv(str(filepath))
        print(f"Saved CSV to: {filepath}")

    # Print stats
    print("\nDataset Statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
