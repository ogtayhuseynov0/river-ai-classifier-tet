"""
Synthetic Data Generator & Classifier Testing UI
Generate synthetic business data and test the AI classifier
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import csv
import io
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from synthetic import SyntheticDataGenerator, GenerationConfig
from main import LeadClassifier, ConversationInput

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Page config
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide"
)

# ============================================================================
# Password Protection (optional - set APP_PASSWORD in .env)
# ============================================================================
def check_password():
    """Simple password protection."""
    app_password = os.getenv("APP_PASSWORD")

    # Skip if no password set
    if not app_password:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Login Required")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False

if not check_password():
    st.stop()

st.title("🧪 Synthetic Data Generator")
st.markdown("Generate synthetic business data and test the AI classifier")

# Initialize classifier
@st.cache_resource
def get_classifier():
    return LeadClassifier()

classifier = get_classifier()

# ============================================================================
# Sidebar - Generation
# ============================================================================

st.sidebar.header("🏭 Generate Data")

# Business prompt
business_prompt = st.sidebar.text_area(
    "Business Description",
    placeholder="e.g., Barbershop in Brooklyn\nLaw firm specializing in immigration\nYacht charter company in Miami",
    height=100,
    help="Describe any business type. AI will generate name, services, and conversations."
)

# Generation settings
st.sidebar.markdown("**Settings**")
num_conversations = st.sidebar.slider("Conversations", 5, 100, 20)
num_contacts = st.sidebar.slider("Contacts", 5, 50, 10)
seed = st.sidebar.number_input("Seed (for reproducibility)", value=42, min_value=0)

# Generate button
if st.sidebar.button("🚀 Generate Dataset", type="primary", use_container_width=True):
    if business_prompt:
        try:
            config = GenerationConfig(
                seed=seed,
                num_conversations=num_conversations,
                num_contacts=num_contacts
            )
            generator = SyntheticDataGenerator()

            # Progress tracking
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()

            def update_progress(step, total, message):
                progress_bar.progress(step / total)
                status_text.text(message)

            with st.spinner("Generating..."):
                dataset = generator.generate(business_prompt, config, progress_callback=update_progress)
                st.session_state.dataset = dataset
                st.session_state.selected_conv_idx = 0

            progress_bar.empty()
            status_text.empty()
            st.sidebar.success(f"Generated {len(dataset.conversations)} conversations!")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.warning("Enter a business description")

# ============================================================================
# Sidebar - Dataset Info & Export
# ============================================================================

if "dataset" in st.session_state:
    dataset = st.session_state.dataset

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dataset")

    stats = dataset.get_stats()
    st.sidebar.markdown(f"**{stats['business_name']}**")
    st.sidebar.caption(stats['business_type'])
    st.sidebar.markdown(f"📝 {stats['total_conversations']} conversations")
    st.sidebar.markdown(f"👥 {stats['total_contacts']} contacts")
    st.sidebar.markdown(f"🛠️ {stats['services_count']} services")

    # Ground truth distribution
    with st.sidebar.expander("Ground Truth Distribution"):
        for gt, count in stats['ground_truth_distribution'].items():
            pct = count / stats['total_conversations'] * 100
            st.markdown(f"• {gt}: {count} ({pct:.0f}%)")

    # Export buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export**")

    # JSON download
    json_data = {
        "metadata": dataset.metadata,
        "business": dataset.business.to_dict(),
        "contacts": [c.to_dict() for c in dataset.contacts],
        "conversations": [c.to_dict() for c in dataset.conversations]
    }
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        "📥 Download JSON",
        data=json_str,
        file_name=f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

    # CSV download
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "conversation_id", "channel", "contact_name", "persona",
        "ground_truth", "scenario_type", "message_count", "messages"
    ])
    for conv in dataset.conversations:
        messages_text = " ||| ".join([
            f"[{m['role'].upper()}] {m['content'][:50]}"
            for m in conv.messages
        ])
        writer.writerow([
            conv.conversation_id, conv.source, conv.contact.get_display_name(),
            conv.contact.persona, conv.ground_truth, conv.scenario_type,
            len(conv.messages), messages_text
        ])
    st.sidebar.download_button(
        "📥 Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# Main Area
# ============================================================================

if "dataset" not in st.session_state:
    # Welcome screen
    st.info("👈 Enter a business description and click **Generate Dataset** to start")

    st.markdown("### Examples")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏥 Medical**")
        st.code("Dental clinic in Los Angeles")
        st.code("Dermatology practice")

    with col2:
        st.markdown("**💼 Professional**")
        st.code("Law firm specializing in immigration")
        st.code("Accounting firm for small businesses")

    with col3:
        st.markdown("**🎨 Services**")
        st.code("Barbershop in Brooklyn")
        st.code("Luxury yacht charter in Miami")

else:
    dataset = st.session_state.dataset

    # Business info header
    st.markdown(f"## {dataset.business.name}")
    st.caption(f"{dataset.business.business_type} • {dataset.business.location}")

    # Services
    with st.expander(f"🛠️ Services ({len(dataset.business.services)})", expanded=False):
        cols = st.columns(3)
        for i, svc in enumerate(dataset.business.services):
            with cols[i % 3]:
                st.markdown(f"**{svc.name}**")
                st.caption(f"{svc.currency} {svc.price}")

    st.markdown("---")

    # Two columns: Chat list and Classification
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 💬 Conversations")

        # Conversation selector
        conv_options = []
        for i, conv in enumerate(dataset.conversations):
            label = f"{conv.source} • {conv.contact.get_display_name()} • {conv.ground_truth}"
            conv_options.append(label)

        selected_idx = st.selectbox(
            "Select conversation",
            range(len(conv_options)),
            format_func=lambda x: conv_options[x],
            key="conv_selector"
        )

        if selected_idx is not None:
            st.session_state.selected_conv_idx = selected_idx

        # Display selected conversation
        conv = dataset.conversations[st.session_state.get("selected_conv_idx", 0)]

        # Conversation metadata
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.metric("Channel", conv.source)
        with meta_cols[1]:
            st.metric("Persona", conv.contact.persona)
        with meta_cols[2]:
            st.metric("Ground Truth", conv.ground_truth)
        with meta_cols[3]:
            st.metric("Messages", len(conv.messages))

        st.markdown("---")

        # Messages
        for msg in conv.messages:
            is_customer = msg["role"] == "customer"
            with st.container(border=True):
                if is_customer:
                    st.markdown("**👤 Customer**")
                else:
                    st.markdown("**🏢 Business**")
                st.markdown(msg["content"])

    with col2:
        st.markdown("### 🎯 Classification")

        # Classify button
        if st.button("🚀 Classify This Conversation", type="primary", use_container_width=True):
            conv = dataset.conversations[st.session_state.get("selected_conv_idx", 0)]

            with st.spinner("Classifying & Extracting..."):
                try:
                    start_time = time.time()

                    conversation_input = ConversationInput(
                        conversation_id=conv.conversation_id,
                        source=conv.source,
                        messages=conv.messages,
                        clinic_name=dataset.business.name,
                        clinic_type=dataset.business.business_type,
                        services=dataset.business.get_service_names()
                    )

                    result = classifier.classify(conversation_input)
                    extracted = classifier.extract(conversation_input)
                    st.session_state.last_result = result
                    st.session_state.last_extracted = extracted
                    st.session_state.last_response_time = time.time() - start_time
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

        # Show ground truth
        conv = dataset.conversations[st.session_state.get("selected_conv_idx", 0)]

        with st.container(border=True):
            st.markdown("**Expected (Ground Truth)**")
            if conv.ground_truth == "lead":
                st.success(f"🟢 LEAD ({conv.scenario_type})")
            elif conv.ground_truth == "not_lead":
                st.error(f"🔴 NOT LEAD ({conv.scenario_type})")
            else:
                st.warning(f"⚪ NEEDS INFO ({conv.scenario_type})")

        # Show AI result
        if st.session_state.get("last_result"):
            result = st.session_state.last_result

            with st.container(border=True):
                st.markdown("**AI Classification**")

                if result.is_lead:
                    st.success(f"🟢 LEAD ({result.confidence:.0%} confidence)")
                elif result.classification == "not_lead":
                    st.error(f"🔴 NOT LEAD ({result.confidence:.0%} confidence)")
                else:
                    st.warning(f"⚪ NEEDS INFO ({result.confidence:.0%} confidence)")

                # Match check
                if conv.ground_truth == result.classification:
                    st.success("✅ Match!")
                else:
                    st.error(f"❌ Mismatch: Expected {conv.ground_truth}, got {result.classification}")

            # Reasoning
            with st.container(border=True):
                st.markdown("**💭 AI Reasoning**")
                st.markdown(result.reasoning)

            # Key signals
            if result.key_signals:
                with st.container(border=True):
                    st.markdown("**🔍 Key Signals**")
                    for signal in result.key_signals:
                        st.markdown(f"• {signal}")

            # Extracted Data
            extracted = st.session_state.get("last_extracted")
            if extracted:
                # Matched Services
                if extracted.matched_services:
                    with st.container(border=True):
                        st.markdown("**🎯 Matched Services**")
                        service_lookup = {s.name: s for s in dataset.business.services}
                        for match in extracted.matched_services:
                            svc_name = match.get("service", "")
                            confidence = match.get("confidence", 0)
                            svc_info = service_lookup.get(svc_name)
                            if svc_info:
                                st.markdown(f"• **{svc_name}** - {svc_info.currency} {svc_info.price} ({confidence:.0%} match)")
                            else:
                                st.markdown(f"• **{svc_name}** ({confidence:.0%} match)")

                with st.container(border=True):
                    st.markdown("**📋 Extracted Data**")
                    fields = [
                        ("Name", f"{extracted.first_name or ''} {extracted.last_name or ''}".strip()),
                        ("DOB", extracted.date_of_birth),
                        ("Gender", extracted.gender),
                        ("City", extracted.city),
                        ("Country", extracted.country),
                        ("Language", extracted.language),
                        ("Occupation", extracted.occupation),
                    ]
                    for label, value in fields:
                        if value:
                            st.markdown(f"• **{label}:** {value}")

                    if extracted.metadata:
                        st.markdown("**Metadata:**")
                        for key, val in extracted.metadata.items():
                            st.markdown(f"  • {key}: {val}")

            # Metrics
            st.markdown("---")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{result.confidence:.0%}")
            with m2:
                st.metric("Response Time", f"{st.session_state.get('last_response_time', 0):.2f}s")

        else:
            st.info("Click **Classify** to see AI result")

        # Contact info
        with st.expander("👤 Contact Details"):
            contact = conv.contact
            st.markdown(f"**Name:** {contact.get_display_name()}")
            st.markdown(f"**Persona:** {contact.persona}")
            if contact.demographics:
                for key, val in contact.demographics.items():
                    st.markdown(f"**{key.title()}:** {val}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {classifier.model_name}")
