"""
Lead Classifier Testing UI - Supabase Connected
Compare AI classification vs human labels to fine-tune prompts
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

from main import LeadClassifier, ConversationInput

# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Page config
st.set_page_config(
    page_title="Lead Classifier - Testing",
    page_icon="🧪",
    layout="wide"
)

# ============================================================================
# Password Protection (optional - set APP_PASSWORD in secrets)
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

# Initialize Supabase
@st.cache_resource
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing SUPABASE_URL or SUPABASE_KEY in .env")
        st.stop()
    return create_client(url, key)

supabase = get_supabase()

# Initialize classifier
@st.cache_resource
def get_classifier():
    return LeadClassifier()

classifier = get_classifier()

# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_organisations():
    """Load all organisations from Supabase."""
    response = supabase.table("organisations").select("id, name").order("name").execute()
    return response.data

@st.cache_data(ttl=300)
def load_org_services(org_id: str):
    """Load services for an organisation."""
    response = supabase.table("organisation_services").select(
        "id, name, description, category_id"
    ).eq("org_id", org_id).execute()
    return response.data

@st.cache_data(ttl=60)
def load_chats(org_id: str, channel_type: str = None, limit: int = 50):
    """Load recent single chats for an organisation (no group chats)."""
    query = supabase.schema("crm").table("chats").select(
        "id, org_id, channel_type, title, last_message_at, last_message_preview, is_archived, is_group"
    ).eq("org_id", org_id).eq("is_archived", False).eq("is_group", False).order("last_message_at", desc=True).limit(limit)

    if channel_type:
        query = query.eq("channel_type", channel_type)

    response = query.execute()
    return response.data

@st.cache_data(ttl=60)
def load_chat_messages(thread_id: str, limit: int = 50):
    """Load messages for a chat thread."""
    response = supabase.schema("crm").table("chat_messages").select(
        "id, direction, body, sent_at, received_at, created_at, contact_id"
    ).eq("thread_id", thread_id).order("created_at", desc=False).limit(limit).execute()
    return response.data

@st.cache_data(ttl=60)
def load_contact(contact_id: str):
    """Load contact details including lifecycle (Lead/Customer)."""
    try:
        response = supabase.schema("crm").table("contacts").select(
            "id, first_name, last_name, display_name, lifecycle, lead_score"
        ).eq("id", contact_id).single().execute()
        return response.data
    except:
        return None

@st.cache_data(ttl=60)
def get_contact_id_from_chat(thread_id: str) -> str | None:
    """Get contact_id from chat_participants (non-self participant)."""
    response = supabase.schema("crm").table("chat_participants").select(
        "contact_id"
    ).eq("thread_id", thread_id).eq("is_self", False).not_.is_(
        "contact_id", "null"
    ).limit(1).execute()

    if response.data and response.data[0].get("contact_id"):
        return response.data[0]["contact_id"]

    return None

def get_contact_from_chat(thread_id: str):
    """Get contact from chat thread."""
    contact_id = get_contact_id_from_chat(thread_id)
    if contact_id:
        return load_contact(contact_id)
    return None

# ============================================================================
# UI
# ============================================================================

st.title("🧪 Lead Classifier Testing")
st.markdown("Compare AI classification vs human labels")

# Sidebar - Organisation Selection
st.sidebar.header("🏥 Organisation")

orgs = load_organisations()
if not orgs:
    st.warning("No organisations found in database")
    st.stop()

org_options = {org["name"]: org["id"] for org in orgs}
selected_org_name = st.sidebar.selectbox("Select Organisation", list(org_options.keys()))
selected_org_id = org_options[selected_org_name]

# Load services for selected org
services = load_org_services(selected_org_id)
service_names = [s["name"] for s in services] if services else ["No services found"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Services ({len(service_names)})**")
with st.sidebar.expander("View Services"):
    for svc in service_names[:10]:
        st.markdown(f"• {svc}")
    if len(service_names) > 10:
        st.caption(f"...and {len(service_names) - 10} more")

# Sidebar - Chat Selection
st.sidebar.markdown("---")
st.sidebar.header("💬 Chats")

# Channel filter
channel_options = ["All", "Instagram", "WhatsApp", "Facebook", "Email", "SMS"]
selected_channel = st.sidebar.selectbox("Channel", channel_options)

# Reset pagination when org or channel changes
filter_key = f"{selected_org_id}_{selected_channel}"
if st.session_state.get("filter_key") != filter_key:
    st.session_state.filter_key = filter_key
    st.session_state.chat_limit = 20

# Pagination state
if "chat_limit" not in st.session_state:
    st.session_state.chat_limit = 20

# Load chats (ordered by last_message_at desc)
channel_filter = None if selected_channel == "All" else selected_channel
chats = load_chats(selected_org_id, channel_type=channel_filter, limit=st.session_state.chat_limit)

if not chats:
    st.info("No chats found for this organisation")
    st.stop()

# Display chat list
st.sidebar.markdown(f"**{len(chats)} chats loaded**")

# Store selected chat in session
if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = None

for chat in chats:
    # Use chat title or preview - NO preloading of messages
    chat_title = chat.get("title") or "Chat"
    preview = (chat.get("last_message_preview") or "")[:25]

    # Channel icon
    channel = chat.get("channel_type", "")
    channel_icon = {"Instagram": "📸", "WhatsApp": "💬", "Facebook": "👤", "Email": "📧", "SMS": "📱"}.get(channel, "💬")

    # Chat button - simple label without contact lookup
    label = f"{channel_icon} {chat_title[:15] if chat_title != 'Chat' else preview[:15] or 'Chat'}"

    if st.sidebar.button(label, key=f"chat_{chat['id']}", use_container_width=True, help=preview):
        st.session_state.selected_chat_id = chat["id"]
        st.session_state.selected_contact = None  # Will load on demand
        st.session_state.last_result = None
        st.rerun()

# Load more button
if len(chats) >= st.session_state.chat_limit:
    if st.sidebar.button("📥 Load more chats", use_container_width=True):
        st.session_state.chat_limit += 20
        st.rerun()

# Main area - two columns
if st.session_state.selected_chat_id:
    col1, col2 = st.columns([1, 1], gap="large")

    # Load messages (only when chat is selected)
    messages = load_chat_messages(st.session_state.selected_chat_id)

    # Load contact on demand (from first message with contact_id)
    if st.session_state.get("selected_contact") is None:
        for msg in messages or []:
            if msg.get("contact_id"):
                st.session_state.selected_contact = load_contact(msg["contact_id"])
                break

    with col1:
        st.markdown("### 💬 Conversation")

        # Classify button at TOP
        if st.button("🚀 Classify Conversation", type="primary", use_container_width=True):
            if messages:
                # Format messages for classifier
                formatted_messages = []
                for msg in messages:
                    if msg.get("body"):
                        formatted_messages.append({
                            "role": "customer" if msg.get("direction") == "INBOUND" else "business",
                            "content": msg.get("body", "")
                        })

                if formatted_messages:
                    with st.spinner("Classifying..."):
                        try:
                            start_time = time.time()

                            conversation = ConversationInput(
                                conversation_id=st.session_state.selected_chat_id,
                                source="chat",
                                messages=formatted_messages,
                                clinic_name=selected_org_name,
                                clinic_type="medical practice",
                                services=service_names
                            )

                            result = classifier.classify(conversation)
                            st.session_state.last_result = result
                            st.session_state.last_response_time = time.time() - start_time
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        st.markdown(f"**{len(messages) if messages else 0} messages**")
        st.markdown("---")

        # Display messages
        if messages:
            for msg in messages:
                is_inbound = msg.get("direction") == "INBOUND"
                body = msg.get("body", "")

                if not body:
                    continue

                with st.container(border=True):
                    if is_inbound:
                        st.markdown("**👤 Customer**")
                    else:
                        st.markdown("**🏥 Business**")
                    st.markdown(body)
        else:
            st.info("No messages in this chat")

    with col2:
        st.markdown("### 🎯 Classification Result")

        # Get actual lifecycle from contact
        contact = st.session_state.get("selected_contact")
        lifecycle = contact.get("lifecycle") if contact else None

        # Show actual status (human label)
        with st.container(border=True):
            st.markdown("**Human Label (lifecycle)**")
            if lifecycle == "Lead":
                st.success("🟢 Lead")
            elif lifecycle == "Customer":
                st.info("🔵 Customer (converted)")
            else:
                st.warning("⚪ Not labeled yet")

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

                # Match indicator
                if lifecycle:
                    human_says_lead = lifecycle == "Lead"
                    ai_says_lead = result.is_lead

                    # Lead vs Lead or Customer vs NOT_LEAD
                    if human_says_lead and ai_says_lead:
                        st.success("✅ Match! Both say Lead")
                    elif not human_says_lead and not ai_says_lead:
                        st.success("✅ Match! Human=Customer, AI=NOT_LEAD")
                    elif human_says_lead and not ai_says_lead:
                        st.error("❌ Mismatch: Human=Lead, AI=NOT_LEAD")
                    else:
                        st.warning("⚠️ Human=Customer but AI=LEAD (already converted?)")

            # Reasoning
            with st.container(border=True):
                st.markdown("**💭 AI Reasoning**")
                st.markdown(result.reasoning)

            # Signals
            if result.key_signals:
                with st.container(border=True):
                    st.markdown("**🔍 Key Signals**")
                    for signal in result.key_signals:
                        st.markdown(f"• {signal}")

            # Metrics
            st.markdown("---")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{result.confidence:.0%}")
            with m2:
                response_time = st.session_state.get("last_response_time", 0)
                st.metric("Response Time", f"{response_time:.2f}s")
        else:
            st.info("Click 'Classify Conversation' to see AI result")

else:
    st.info("👈 Select a chat from the sidebar to start testing")

    # Legend
    st.markdown("### Legend")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Lead** - potential customer")
    with col2:
        st.markdown("🔵 **Customer** - converted")
    with col3:
        st.markdown("⚪ **Unknown** - not labeled")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("**Legend:** 🟢 Lead | 🔵 Customer | ⚪ Unknown")
st.sidebar.caption(f"Org: {selected_org_id[:8]}...")
st.sidebar.caption(f"Model: {classifier.model_name}")
