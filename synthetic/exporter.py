"""Export utilities for synthetic datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SyntheticDataset


def export_to_json(dataset: SyntheticDataset, filepath: str, indent: int = 2) -> None:
    """
    Export dataset to JSON file.

    Args:
        dataset: The dataset to export
        filepath: Output file path
        indent: JSON indentation (default: 2)
    """
    data = {
        "metadata": dataset.metadata,
        "business": dataset.business.to_dict(),
        "contacts": [c.to_dict() for c in dataset.contacts],
        "conversations": [c.to_dict() for c in dataset.conversations]
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_to_csv(dataset: SyntheticDataset, filepath: str) -> None:
    """
    Export conversations to CSV file (flat format).

    Args:
        dataset: The dataset to export
        filepath: Output file path
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "conversation_id",
            "business_name",
            "business_type",
            "channel",
            "contact_name",
            "persona",
            "ground_truth",
            "scenario_type",
            "message_count",
            "first_message",
            "last_message",
            "all_messages"
        ])

        # Rows
        for conv in dataset.conversations:
            first_msg = conv.messages[0]["content"][:100] if conv.messages else ""
            last_msg = conv.messages[-1]["content"][:100] if conv.messages else ""

            # Format all messages
            all_msgs = " ||| ".join(
                f"[{m['role'].upper()}] {m['content'][:100]}"
                for m in conv.messages
            )

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
                last_msg,
                all_msgs
            ])


def export_for_classifier(dataset: SyntheticDataset, filepath: str) -> None:
    """
    Export dataset in format ready for classifier testing.

    Creates a JSON file with conversation inputs and expected results.

    Args:
        dataset: The dataset to export
        filepath: Output file path
    """
    test_cases = []

    for conv in dataset.conversations:
        test_cases.append({
            "conversation_input": {
                "conversation_id": conv.conversation_id,
                "messages": conv.messages,
                "source": conv.source,
                "clinic_name": conv.business.name,
                "clinic_type": conv.business.business_type,
                "services": conv.business.get_service_names()
            },
            "expected": {
                "ground_truth": conv.ground_truth,
                "scenario_type": conv.scenario_type,
                "contact_persona": conv.contact.persona
            }
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)


def load_dataset(filepath: str) -> dict:
    """
    Load a dataset from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dataset dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_stats(dataset: SyntheticDataset) -> None:
    """Print dataset statistics to console."""
    stats = dataset.get_stats()

    print("\n" + "=" * 50)
    print("SYNTHETIC DATASET STATISTICS")
    print("=" * 50)
    print(f"Business: {stats['business_name']} ({stats['business_type']})")
    print(f"Services: {stats['services_count']}")
    print(f"Contacts: {stats['total_contacts']}")
    print(f"Conversations: {stats['total_conversations']}")
    print()
    print("Ground Truth Distribution:")
    for gt, count in stats["ground_truth_distribution"].items():
        pct = count / stats["total_conversations"] * 100
        print(f"  {gt}: {count} ({pct:.1f}%)")
    print()
    print("Channel Distribution:")
    for channel, count in stats["channel_distribution"].items():
        pct = count / stats["total_conversations"] * 100
        print(f"  {channel}: {count} ({pct:.1f}%)")
    print()
    print("Persona Distribution:")
    for persona, count in stats["persona_distribution"].items():
        pct = count / stats["total_conversations"] * 100
        print(f"  {persona}: {count} ({pct:.1f}%)")
    print("=" * 50 + "\n")
