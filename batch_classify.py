"""
Batch Classification Script
Generates Excel report with classification results for multiple organizations
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from main import LeadClassifier, ConversationInput, ExtractedData

# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ============================================================================
# Configuration
# ============================================================================

# Organization IDs to process
ORG_IDS = [
    "org_34Idi1MhESIDmZYFbS6ExbT2NyN",
    # "org_33m9zcPA8Ly0bwQTTRl6CdxFLsI",
]

# How many chats per organization (set to None for all)
CHAT_LIMIT = 500

# Run only on chats with lifecycle marked (Lead/Customer), or all chats
RUN_ONLY_MARKED = True

# Output file
OUTPUT_FILE = f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# ============================================================================
# Initialize clients
# ============================================================================

def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise Exception("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

supabase = get_supabase()
classifier = LeadClassifier()

# ============================================================================
# Data fetching functions
# ============================================================================

def fetch_organisation(org_id: str) -> dict:
    """Fetch organisation details."""
    response = supabase.table("organisations").select(
        "id, name"
    ).eq("id", org_id).single().execute()
    return response.data

def fetch_org_services(org_id: str) -> list[str]:
    """Fetch services for an organisation."""
    response = supabase.table("organisation_services").select(
        "name"
    ).eq("org_id", org_id).execute()
    return [s["name"] for s in response.data] if response.data else []

def fetch_chats(org_id: str, limit: int = None) -> list[dict]:
    """Fetch chats for an organisation (single chats only, ordered by last message)."""
    query = supabase.schema("crm").table("chats").select(
        "id, org_id, channel_type, title, last_message_at, last_message_preview"
    ).eq("org_id", org_id).eq("is_archived", False).eq("is_group", False).order(
        "last_message_at", desc=True
    )

    if limit:
        query = query.limit(limit)

    response = query.execute()
    return response.data or []

def fetch_chat_messages(thread_id: str, limit: int = 100) -> list[dict]:
    """Fetch messages for a chat."""
    response = supabase.schema("crm").table("chat_messages").select(
        "id, direction, body, sent_at, received_at, created_at, contact_id"
    ).eq("thread_id", thread_id).order("created_at", desc=False).limit(limit).execute()
    return response.data or []

def fetch_contact(contact_id: str) -> dict:
    """Fetch contact details."""
    try:
        response = supabase.schema("crm").table("contacts").select(
            "id, first_name, last_name, display_name, lifecycle"
        ).eq("id", contact_id).single().execute()
        return response.data
    except:
        return None

def fetch_chats_with_contacts(org_id: str, limit: int = None, only_marked: bool = True) -> list[dict]:
    """
    Fetch chats with contact info in bulk (3 queries instead of N*2).
    Returns list of dicts with: chat_id, channel_type, last_message_at, contact_name, lifecycle
    """
    # 1. Get all chats for org
    query = supabase.schema("crm").table("chats").select(
        "id, channel_type, last_message_at"
    ).eq("org_id", org_id).eq("is_archived", False).eq("is_group", False).order(
        "last_message_at", desc=True
    )
    if limit:
        query = query.limit(limit)
    chats = query.execute().data or []

    if not chats:
        return []

    chat_ids = [c["id"] for c in chats]

    # 2. Get all participants (non-self) for these chats
    participants = supabase.schema("crm").table("chat_participants").select(
        "thread_id, contact_id, display_name"
    ).in_("thread_id", chat_ids).eq("is_self", False).not_.is_(
        "contact_id", "null"
    ).execute().data or []

    # Map thread_id -> participant info
    thread_to_participant = {}
    for p in participants:
        if p["thread_id"] not in thread_to_participant:
            thread_to_participant[p["thread_id"]] = p

    # 3. Get all contacts for these contact_ids
    contact_ids = list(set(p["contact_id"] for p in participants if p.get("contact_id")))

    contacts_map = {}
    if contact_ids:
        contacts = supabase.schema("crm").table("contacts").select(
            "id, first_name, last_name, display_name, lifecycle"
        ).in_("id", contact_ids).execute().data or []
        contacts_map = {c["id"]: c for c in contacts}

    # 4. Join and filter
    result = []
    for chat in chats:
        chat_id = chat["id"]
        participant = thread_to_participant.get(chat_id)

        contact_name = "Unknown"
        lifecycle = None

        if participant:
            contact_id = participant.get("contact_id")
            if contact_id and contact_id in contacts_map:
                contact = contacts_map[contact_id]
                contact_name = contact.get("display_name") or f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip() or participant.get("display_name", "Unknown")
                lifecycle = contact.get("lifecycle")
            else:
                contact_name = participant.get("display_name", "Unknown")

        # Filter by lifecycle if only_marked
        if only_marked and lifecycle not in ["Lead", "Customer"]:
            continue

        result.append({
            "chat_id": chat_id,
            "channel_type": chat.get("channel_type", "Unknown"),
            "last_message_at": chat.get("last_message_at", ""),
            "contact_name": contact_name,
            "lifecycle": lifecycle
        })

    return result

# ============================================================================
# Classification logic
# ============================================================================

def classify_chat(chat_info: dict, org_name: str, services: list[str]) -> dict | None:
    """Classify a single chat and return result row.

    Args:
        chat_info: Pre-fetched chat data with contact info (from fetch_chats_with_contacts)
        org_name: Organization name
        services: List of services

    Returns:
        Result dict or None if skipped
    """

    chat_id = chat_info["chat_id"]
    channel_type = chat_info.get("channel_type", "Unknown")
    contact_name = chat_info.get("contact_name", "Unknown")
    ground_truth = chat_info.get("lifecycle")  # Already filtered if RUN_ONLY_MARKED

    # Now fetch messages (only for chats we'll actually process)
    messages = fetch_chat_messages(chat_id)
    if not messages:
        return None  # Skip chats without messages

    # Format messages for classifier
    formatted_messages = []
    for msg in messages:
        if msg.get("body"):
            formatted_messages.append({
                "role": "customer" if msg.get("direction") == "INBOUND" else "business",
                "content": msg.get("body", "")
            })

    if not formatted_messages:
        return None  # Skip - no message content

    # Format last 5 messages for Excel (most recent last)
    last_5 = messages[-5:] if len(messages) >= 5 else messages
    last_5_formatted = " ||| ".join([
        f"[{msg.get('direction', '?')}] {msg.get('body', '')[:100]}"
        for msg in last_5 if msg.get("body")
    ])

    # Run classification
    try:
        conversation = ConversationInput(
            conversation_id=chat_id,
            source=channel_type.lower() if channel_type else "chat",
            messages=formatted_messages,
            clinic_name=org_name,
            clinic_type="medical practice",
            services=services
        )

        result = classifier.classify(conversation)
        extracted = classifier.extract(conversation)

        # Determine match
        ai_class = result.classification.upper()

        # Handle NEEDS_INFO separately - it's not a classification decision
        if ai_class == "NEEDS_INFO":
            match = "⚠ Needs Info"
        elif ground_truth == "Lead":
            if ai_class == "LEAD":
                match = "✓ Match"
            else:
                match = "✗ Mismatch"
        elif ground_truth == "Customer":
            if ai_class == "NOT_LEAD":
                match = "✓ Match"
            elif ai_class == "LEAD":
                match = "⚠ Customer (was lead)"
            else:
                match = "✗ Mismatch"
        else:
            match = "Unknown"

        return {
            "clinic_name": org_name,
            "chat_id": chat_id,
            "message_count": len(messages),
            "channel": channel_type,
            "contact_name": contact_name,
            "last_message_at": chat_info.get("last_message_at", ""),
            "ground_truth": ground_truth or "Unknown",
            "ai_classification": ai_class,
            "confidence": round(result.confidence, 2),
            "match": match,
            "reasoning": result.reasoning,
            "key_signals": ", ".join(result.key_signals) if result.key_signals else "",
            "last_5_messages": last_5_formatted,
            # Extracted data
            "ext_first_name": extracted.first_name,
            "ext_last_name": extracted.last_name,
            "ext_date_of_birth": extracted.date_of_birth,
            "ext_gender": extracted.gender,
            "ext_city": extracted.city,
            "ext_country": extracted.country,
            "ext_language": extracted.language,
            "ext_occupation": extracted.occupation,
            "ext_matched_services": ", ".join([s.get("service", "") for s in extracted.matched_services]) if extracted.matched_services else "",
            "ext_metadata": json.dumps(extracted.metadata) if extracted.metadata else "",
            "note": ""
        }

    except Exception as e:
        return {
            "clinic_name": org_name,
            "chat_id": chat_id,
            "channel": channel_type,
            "contact_name": contact_name,
            "last_message_at": chat_info.get("last_message_at", ""),
            "ground_truth": ground_truth or "Unknown",
            "ai_classification": "ERROR",
            "confidence": 0,
            "match": "Error",
            "reasoning": str(e)[:200],
            "key_signals": "",
            "last_5_messages": last_5_formatted,
            "message_count": len(messages) if messages else 0,
            # Extracted data (empty on error)
            "ext_first_name": None,
            "ext_last_name": None,
            "ext_date_of_birth": None,
            "ext_gender": None,
            "ext_city": None,
            "ext_country": None,
            "ext_language": None,
            "ext_occupation": None,
            "ext_matched_services": "",
            "ext_metadata": "",
            "note": ""
        }

# ============================================================================
# Main script
# ============================================================================

def process_organisation(org_id: str) -> pd.DataFrame:
    """Process all chats for an organisation and return DataFrame."""

    print(f"\n{'='*60}")
    print(f"Processing organisation: {org_id}")
    print(f"{'='*60}")

    # Fetch org details
    org = fetch_organisation(org_id)
    if not org:
        print(f"  ❌ Organisation not found: {org_id}")
        return None

    org_name = org.get("name", "Unknown")
    print(f"  📍 Organisation: {org_name}")

    # Fetch services
    services = fetch_org_services(org_id)
    print(f"  📋 Services: {len(services)}")

    # Fetch chats with contact info (bulk query)
    print(f"  🔍 Fetching chats (only_marked={RUN_ONLY_MARKED})...")
    chats = fetch_chats_with_contacts(org_id, limit=CHAT_LIMIT, only_marked=RUN_ONLY_MARKED)
    print(f"  💬 Chats to process: {len(chats)}")

    if not chats:
        print(f"  ⚠️ No chats found" + (" with ground truth" if RUN_ONLY_MARKED else ""))
        return None

    # Process each chat
    results = []
    processed = 0

    for i, chat_info in enumerate(chats, 1):
        print(f"  [{i}/{len(chats)}] Classifying {chat_info['chat_id'][:8]}...", end=" ")

        start_time = time.time()
        row = classify_chat(chat_info, org_name, services)
        elapsed = time.time() - start_time

        if row is None:
            print(f"⏭ Skipped (no messages)")
            continue

        processed += 1
        results.append(row)
        print(f"→ {row['ai_classification']} ({row['confidence']}) [{elapsed:.1f}s]")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    if not results:
        print(f"  ⚠️ No chats processed")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    columns = [
        "clinic_name", "chat_id", "channel", "contact_name", "last_message_at", "ground_truth",
        "ai_classification", "confidence", "match",
        "reasoning", "key_signals", "last_5_messages", "message_count",
        # Extracted data
        "ext_first_name", "ext_last_name", "ext_date_of_birth", "ext_gender",
        "ext_city", "ext_country", "ext_language", "ext_occupation",
        "ext_matched_services", "ext_metadata",
        "note"
    ]
    df = df[columns]

    # Sort by last_message_at (most recent first)
    df = df.sort_values("last_message_at", ascending=False)
    df = df.reset_index(drop=True)

    # Summary stats
    total = len(df)
    matches = len(df[df["match"] == "✓ Match"])
    mismatches = len(df[df["match"] == "✗ Mismatch"])
    customer_leads = len(df[df["match"] == "⚠ Customer (was lead)"])
    needs_info = len(df[df["match"] == "⚠ Needs Info"])
    # Accuracy excludes needs_info (undecided)
    decided = total - needs_info
    accuracy = matches / decided * 100 if decided > 0 else 0

    print(f"\n  📊 Summary:")
    print(f"     Processed: {processed}")
    print(f"     Matches: {matches} ({accuracy:.1f}%)")
    print(f"     Mismatches: {mismatches}")
    print(f"     Customer (was lead): {customer_leads}")
    print(f"     Needs Info: {needs_info}")

    # Return df and summary dict
    summary = {
        "clinic_name": org_name,
        "total_chats": len(chats),
        "processed": processed,
        "matches": matches,
        "mismatches": mismatches,
        "customer_was_lead": customer_leads,
        "needs_info": needs_info,
        "accuracy_pct": round(accuracy, 1)
    }

    return df, summary

def main():
    """Main entry point."""

    if not ORG_IDS:
        print("❌ No organization IDs configured!")
        print("   Edit ORG_IDS list in this script to add organization IDs.")
        return

    print(f"\n🚀 Batch Classification Script")
    print(f"   Organizations: {len(ORG_IDS)}")
    print(f"   Chat limit per org: {CHAT_LIMIT or 'All'}")
    print(f"   Output: {OUTPUT_FILE}")

    # Process each organisation
    org_dataframes = {}
    summaries = []

    for org_id in ORG_IDS:
        result = process_organisation(org_id)
        if result is not None:
            df, summary = result
            # Get org name for sheet name
            org = fetch_organisation(org_id)
            sheet_name = (org.get("name", org_id) if org else org_id)[:31]  # Excel sheet name limit
            org_dataframes[sheet_name] = df
            summaries.append(summary)

    # Write to Excel with multiple sheets
    if org_dataframes:
        print(f"\n{'='*60}")
        print(f"📁 Writing to {OUTPUT_FILE}")

        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            # Summary sheet first
            if summaries:
                summary_df = pd.DataFrame(summaries)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Auto-adjust summary column widths
                worksheet = writer.sheets["Summary"]
                for i, col in enumerate(summary_df.columns):
                    max_length = max(
                        summary_df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    worksheet.column_dimensions[chr(65 + i)].width = min(max_length, 30)

            # Individual org sheets
            for sheet_name, df in org_dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    worksheet.column_dimensions[chr(65 + i)].width = min(max_length, 50)

        print(f"✅ Done! Report saved to: {OUTPUT_FILE}")
    else:
        print("\n❌ No data to export")

if __name__ == "__main__":
    main()
