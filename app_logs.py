"""
AI Logs Viewer - Streamlit App
View classification logs and draft messages per organization
"""

import os
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# =============================================================================
# Password Protection
# =============================================================================

def check_password():
    """Simple password protection with rate limiting."""
    app_password = os.getenv("APP_PASSWORD")

    # Skip if no password set
    if not app_password:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "lockout_until" not in st.session_state:
        st.session_state.lockout_until = None

    if st.session_state.authenticated:
        return True

    # Check if locked out
    if st.session_state.lockout_until:
        if datetime.now() < st.session_state.lockout_until:
            remaining = (st.session_state.lockout_until - datetime.now()).seconds
            st.error(f"Too many attempts. Try again in {remaining} seconds.")
            return False
        else:
            # Lockout expired, reset
            st.session_state.login_attempts = 0
            st.session_state.lockout_until = None

    st.title("🔐 Login Required")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == app_password:
            st.session_state.authenticated = True
            st.session_state.login_attempts = 0
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            if st.session_state.login_attempts >= 5:
                st.session_state.lockout_until = datetime.now() + timedelta(minutes=3)
                st.error("Too many attempts. Locked for 3 minutes.")
            else:
                remaining = 5 - st.session_state.login_attempts
                st.error(f"Incorrect password. {remaining} attempts remaining.")

    return False

if not check_password():
    st.stop()

# Page config
st.set_page_config(
    page_title="AI Logs Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Database Connections
# =============================================================================

@st.cache_resource
def get_crm_db():
    """Get CRM database client (read-only)."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None

@st.cache_resource
def get_results_db():
    """Get Results database client (read/write for notes)."""
    url = os.getenv("RESULTS_SUPABASE_URL")
    key = os.getenv("RESULTS_SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None

crm_db = get_crm_db()
results_db = get_results_db()

# =============================================================================
# Data Loading Functions (CRM DB)
# =============================================================================

@st.cache_data(ttl=300)
def load_organisations():
    """Load all organisations."""
    if not crm_db:
        return []
    try:
        response = crm_db.table("organisations").select("id, name").order("name").execute()
        return response.data or []
    except Exception as e:
        st.error(f"Error loading organisations: {e}")
        return []

def load_chats_with_logs(org_id: str, date_from=None, date_to=None, channel=None, limit=20, offset=0):
    """Load chats that have classification logs, ordered by latest message."""
    if not crm_db or not org_id:
        return [], 0
    try:
        # Get thread_ids that have classification logs
        query = crm_db.schema("crm").table("classification_logs").select("thread_id").eq("org_id", org_id)

        if date_from:
            query = query.gte("created_at", date_from.isoformat())
        if date_to:
            query = query.lte("created_at", date_to.isoformat())

        logs_response = query.execute()
        thread_ids = list(set(log["thread_id"] for log in (logs_response.data or []) if log.get("thread_id")))

        if not thread_ids:
            return [], 0

        # Get chat details ordered by last_message_at
        chats_query = crm_db.schema("crm").table("chats").select("*", count="exact").in_("id", thread_ids)

        if channel:
            chats_query = chats_query.ilike("channel_type", channel)

        # Order by last_message_at (most recent first), fallback to updated_at
        # First get count, then adjust offset if needed
        if offset >= len(thread_ids):
            offset = 0
        chats_response = chats_query.order("last_message_at", desc=True, nullsfirst=False).range(offset, offset + limit - 1).execute()
        total_count = chats_response.count or len(thread_ids)
        return chats_response.data or [], total_count
    except Exception as e:
        st.error(f"Error loading chats: {e}")
        return [], 0

def load_chat_messages(thread_id: str, limit=20, offset=0):
    """Load messages for a chat with pagination (newest first)."""
    if not crm_db or not thread_id:
        return [], 0
    try:
        response = crm_db.schema("crm").table("chat_messages").select("*", count="exact").eq("thread_id", thread_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        total_count = response.count or 0
        return response.data or [], total_count
    except Exception as e:
        st.error(f"Error loading messages: {e}")
        return [], 0

def load_classification_logs(thread_id: str):
    """Load classification logs keyed by message_id."""
    if not crm_db or not thread_id:
        return {}
    try:
        response = crm_db.schema("crm").table("classification_logs").select("*").eq("thread_id", thread_id).execute()
        # Key by message_id for easy lookup
        return {log["message_id"]: log for log in (response.data or []) if log.get("message_id")}
    except Exception as e:
        st.error(f"Error loading classification logs: {e}")
        return {}

def load_draft_messages(thread_id: str):
    """Load draft messages keyed by message_id."""
    if not crm_db or not thread_id:
        return {}
    try:
        response = crm_db.schema("crm").table("draft_messages").select("*").eq("thread_id", thread_id).execute()
        # Key by message_id for easy lookup
        return {draft["message_id"]: draft for draft in (response.data or []) if draft.get("message_id")}
    except Exception as e:
        st.error(f"Error loading draft messages: {e}")
        return {}

# =============================================================================
# Notes Functions (Results DB)
# =============================================================================

def load_notes_for_thread(thread_id: str):
    """Load all notes for a thread, keyed by (log_type, log_id)."""
    if not results_db or not thread_id:
        return {}
    try:
        response = results_db.table("ai_log_notes").select("*").eq("thread_id", str(thread_id)).execute()
        # Key by (log_type, log_id) tuple
        return {(n["log_type"], str(n["log_id"])): n for n in (response.data or [])}
    except Exception as e:
        st.error(f"Error loading notes: {e}")
        return {}

def save_note(log_type: str, log_id: str, thread_id: str, message_id: str, org_id: str, note: str):
    """Save or update a note."""
    if not results_db:
        return False
    try:
        data = {
            "org_id": org_id,
            "log_type": log_type,
            "log_id": log_id,
            "thread_id": thread_id,
            "message_id": message_id,
            "note": note
        }
        # Upsert based on unique constraint
        results_db.table("ai_log_notes").upsert(data, on_conflict="org_id,log_type,log_id").execute()
        return True
    except Exception as e:
        st.error(f"Error saving note: {e}")
        return False

# =============================================================================
# UI Components
# =============================================================================

def render_classification_badge(classification: str, confidence: float):
    """Render classification badge with color."""
    colors = {
        "lead": ("🟢", "#4CAF50"),
        "not_lead": ("🔴", "#f44336"),
        "needs_info": ("🟡", "#FFC107")
    }
    icon, color = colors.get(classification, ("⚪", "#9E9E9E"))
    return f"{icon} **{classification.upper()}** ({confidence:.0%})"

def render_message_card(msg: dict, is_customer: bool):
    """Render a message card."""
    icon = "👤" if is_customer else "🏥"
    label = "Customer" if is_customer else "Business"
    bg_color = "#e3f2fd" if is_customer else "#e8f5e9"
    border_color = "#1976d2" if is_customer else "#388e3c"
    text_color = "#1a1a1a"

    created_at = msg.get("created_at", "")
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M")
        except:
            time_str = ""
    else:
        time_str = ""

    content = msg.get("content", msg.get("body", ""))[:500]

    st.markdown(f"""
    <div style="background: {bg_color}; padding: 10px; border-radius: 8px; margin: 5px 0; color: {text_color}; border-left: 3px solid {border_color};">
        <small style="color: #555;">{icon} <b>{label}</b> · {time_str}</small><br/>
        <span style="color: {text_color};">{content}</span>
    </div>
    """, unsafe_allow_html=True)

def render_classification_card(log: dict, notes: dict, org_id: str, thread_id: str):
    """Render classification log card."""
    if not log:
        st.markdown("*No classification*")
        return

    log_id = str(log.get("id", ""))
    note_key = ("classification", log_id)
    existing_note = notes.get(note_key, {}).get("note", "")

    # Classification badge
    st.markdown(render_classification_badge(
        log.get("classification", "unknown"),
        log.get("confidence", 0)
    ))

    # Reasoning
    if log.get("reasoning"):
        with st.expander("💭 Reasoning"):
            st.write(log["reasoning"])

    # Key signals
    if log.get("key_signals"):
        with st.expander("🎯 Key Signals"):
            signals = log["key_signals"]
            if isinstance(signals, list):
                for s in signals:
                    st.markdown(f"- {s}")
            else:
                st.write(signals)

    # Extracted data
    if log.get("extracted_data"):
        with st.expander("📊 Extracted Data"):
            st.json(log["extracted_data"])

    # Matched services
    if log.get("matched_services"):
        with st.expander("🔧 Matched Services"):
            services = log["matched_services"]
            if isinstance(services, list):
                for s in services:
                    st.markdown(f"- {s}")
            else:
                st.write(services)

    # Meta info
    st.caption(f"Model: {log.get('model_name', 'N/A')} | {log.get('processing_time_ms', 0)}ms")

    # Note field
    note_input = st.text_area(
        "Note",
        value=existing_note,
        key=f"note_class_{log_id}",
        height=60,
        label_visibility="collapsed",
        placeholder="Add note..."
    )
    if st.button("Save", key=f"save_class_{log_id}", use_container_width=True):
        if save_note("classification", log_id, thread_id, log.get("message_id"), org_id, note_input):
            st.success("Saved!")
            st.rerun()

def render_draft_card(draft: dict, notes: dict, org_id: str, thread_id: str):
    """Render draft message card."""
    if not draft:
        st.markdown("*No draft*")
        return

    draft_id = str(draft.get("id", ""))
    note_key = ("draft", draft_id)
    existing_note = notes.get(note_key, {}).get("note", "")

    # Draft content
    st.markdown(f"**Draft Response:**")
    st.info(draft.get("draft_content", ""))

    # Confidence and status
    confidence = draft.get("confidence", 0)
    status = draft.get("status", "unknown")
    st.markdown(f"Confidence: **{confidence:.0%}** | Status: **{status}**")

    # Context snapshot
    if draft.get("context_snapshot"):
        with st.expander("📋 Context Snapshot"):
            st.json(draft["context_snapshot"])

    # Meta info
    st.caption(f"Model: {draft.get('model_name', 'N/A')} | {draft.get('processing_time_ms', 0)}ms")

    # Note field
    note_input = st.text_area(
        "Note",
        value=existing_note,
        key=f"note_draft_{draft_id}",
        height=60,
        label_visibility="collapsed",
        placeholder="Add note..."
    )
    if st.button("Save", key=f"save_draft_{draft_id}", use_container_width=True):
        if save_note("draft", draft_id, thread_id, draft.get("message_id"), org_id, note_input):
            st.success("Saved!")
            st.rerun()

# =============================================================================
# Main App
# =============================================================================

def main():
    st.title("🔍 AI Logs Viewer")

    # Check database connections
    if not crm_db:
        st.error("CRM database not configured. Set SUPABASE_URL and SUPABASE_KEY.")
        return
    if not results_db:
        st.warning("Results database not configured. Notes will not be saved. Set RESULTS_SUPABASE_URL and RESULTS_SUPABASE_KEY.")

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        # Organisation selector
        orgs = load_organisations()
        org_options = {org["name"]: org["id"] for org in orgs}

        if not org_options:
            st.warning("No organisations found")
            return

        selected_org_name = st.selectbox("Organisation", list(org_options.keys()))
        selected_org_id = org_options[selected_org_name]

        st.divider()

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", value=datetime.now() - timedelta(days=30))
        with col2:
            date_to = st.date_input("To", value=datetime.now())

        # Channel filter
        channel = st.selectbox("Channel", ["All", "Email", "WhatsApp", "Telegram", "Instagram", "SMS"])
        channel_filter = None if channel == "All" else channel

        st.divider()

        # Initialize chat pagination
        if "chat_offset" not in st.session_state:
            st.session_state.chat_offset = 0

        # Reset offset when filters change
        filter_key = f"{selected_org_id}_{date_from}_{date_to}_{channel_filter}"
        if st.session_state.get("last_filter_key") != filter_key:
            st.session_state.chat_offset = 0
            st.session_state.last_filter_key = filter_key

        # Load chats
        chats, total_chats = load_chats_with_logs(
            selected_org_id,
            date_from=datetime.combine(date_from, datetime.min.time()),
            date_to=datetime.combine(date_to, datetime.max.time()),
            channel=channel_filter,
            limit=20,
            offset=st.session_state.chat_offset
        )

        st.subheader(f"Chats ({total_chats})")

        # Chat list
        selected_chat = None
        for chat in chats:
            chat_id = chat.get("id")
            chat_title = chat.get("title") or chat.get("contact_name") or chat_id[:8]
            channel_icon = {"instagram": "📸", "whatsapp": "💬", "email": "📧", "sms": "📱"}.get(chat.get("channel"), "💭")

            if st.button(f"{channel_icon} {chat_title}", key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.selected_chat_id = chat_id
                st.session_state.msg_offset = 0  # Reset message pagination

        # Load more chats button
        if st.session_state.chat_offset + len(chats) < total_chats:
            if st.button("Load more chats...", key="load_more_chats"):
                st.session_state.chat_offset += 20
                st.rerun()

        # Set selected chat from session state
        if "selected_chat_id" in st.session_state:
            selected_chat = next((c for c in chats if c["id"] == st.session_state.selected_chat_id), None)

    # Main area
    if not selected_chat:
        st.info("👈 Select a chat from the sidebar to view AI logs")
        return

    thread_id = selected_chat["id"]

    # Initialize message pagination
    if "msg_offset" not in st.session_state:
        st.session_state.msg_offset = 0

    # Load data
    msg_limit = 20
    messages, total_messages = load_chat_messages(thread_id, limit=msg_limit + st.session_state.msg_offset, offset=0)
    classification_logs = load_classification_logs(thread_id)
    draft_messages = load_draft_messages(thread_id)
    notes = load_notes_for_thread(thread_id) if results_db else {}

    # Header
    chat_title = selected_chat.get("title") or selected_chat.get("contact_name") or "Chat"
    st.subheader(f"💬 {chat_title}")
    st.caption(f"Thread: {thread_id} | Messages: {len(messages)}/{total_messages} | Classifications: {len(classification_logs)} | Drafts: {len(draft_messages)}")

    st.divider()

    # Column headers
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.markdown("**Message**")
    with col2:
        st.markdown("**Classification**")
    with col3:
        st.markdown("**Draft Response**")

    st.divider()

    # Messages with side-by-side logs
    for msg in messages:
        msg_id = msg.get("id")
        direction = (msg.get("direction") or "").upper()
        is_customer = direction == "INBOUND"

        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            render_message_card(msg, is_customer)

        with col2:
            log = classification_logs.get(msg_id)
            if log:
                render_classification_card(log, notes, selected_org_id, thread_id)
            else:
                st.markdown("—")

        with col3:
            draft = draft_messages.get(msg_id)
            if draft:
                render_draft_card(draft, notes, selected_org_id, thread_id)
            else:
                st.markdown("—")

        st.divider()

    # Load more messages button
    if len(messages) < total_messages:
        if st.button("Load more messages...", key="load_more_msgs", use_container_width=True):
            st.session_state.msg_offset += 20
            st.rerun()

if __name__ == "__main__":
    main()
