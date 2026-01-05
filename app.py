"""
Streamlit UI for Lead Classifier Testing
"""

import streamlit as st
import time
from main import LeadClassifier, ConversationInput

# Page config
st.set_page_config(
    page_title="Lead Classifier",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Lead Classifier")
st.markdown("Test the AI conversation classifier")

# Model selection
model_options = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)

# Initialize classifier
classifier = LeadClassifier(model_name=selected_model)

# Debug: show API config
st.sidebar.markdown("---")
st.sidebar.caption(f"Mode: {classifier.mode}")
st.sidebar.caption(f"Model: {classifier.model_name}")

# Initialize clinics in session state
if "clinics" not in st.session_state:
    st.session_state.clinics = {
        "Bright Smile Dental": {
            "clinic_type": "dental clinic",
            "services": [
                "Teeth Whitening",
                "Dental Implants",
                "Root Canal Treatment",
                "Invisalign",
                "Regular Checkups",
                "Emergency Dental Care"
            ]
        },
        "City Health Medical Center": {
            "clinic_type": "general medical practice",
            "services": [
                "General Consultation",
                "Annual Physical Exams",
                "Vaccinations",
                "Lab Tests",
                "Chronic Disease Management"
            ]
        },
        "Glow Dermatology": {
            "clinic_type": "dermatology clinic",
            "services": [
                "Acne Treatment",
                "Skin Cancer Screening",
                "Botox & Fillers",
                "Laser Hair Removal",
                "Chemical Peels"
            ]
        }
    }

if "selected_clinic" not in st.session_state:
    st.session_state.selected_clinic = list(st.session_state.clinics.keys())[0]

# Sidebar - Clinic Selection & Configuration
st.sidebar.header("🏥 Clinic Selection")

# Select existing clinic
clinic_names = list(st.session_state.clinics.keys())
selected_clinic = st.sidebar.selectbox(
    "Select Clinic",
    clinic_names,
    index=clinic_names.index(st.session_state.selected_clinic) if st.session_state.selected_clinic in clinic_names else 0
)
st.session_state.selected_clinic = selected_clinic

# Get current clinic data
current_clinic = st.session_state.clinics[selected_clinic]

st.sidebar.markdown("---")
st.sidebar.header("✏️ Edit Clinic")

# Edit clinic details
with st.sidebar.expander("Edit Current Clinic", expanded=False):
    new_clinic_name = st.text_input("Clinic Name", value=selected_clinic, key="edit_name")
    new_clinic_type = st.text_input("Clinic Type", value=current_clinic["clinic_type"], key="edit_type")
    new_services_text = st.text_area(
        "Services (one per line)",
        value="\n".join(current_clinic["services"]),
        key="edit_services"
    )

    if st.button("💾 Save Changes"):
        new_services = [s.strip() for s in new_services_text.split("\n") if s.strip()]

        # If name changed, update key
        if new_clinic_name != selected_clinic:
            del st.session_state.clinics[selected_clinic]

        st.session_state.clinics[new_clinic_name] = {
            "clinic_type": new_clinic_type,
            "services": new_services
        }
        st.session_state.selected_clinic = new_clinic_name
        st.success("Saved!")
        st.rerun()

# Add new clinic
with st.sidebar.expander("➕ Add New Clinic", expanded=False):
    add_clinic_name = st.text_input("New Clinic Name", key="add_name")
    add_clinic_type = st.text_input("Clinic Type", value="medical practice", key="add_type")
    add_services_text = st.text_area(
        "Services (one per line)",
        value="Service 1\nService 2\nService 3",
        key="add_services"
    )

    if st.button("➕ Add Clinic"):
        if add_clinic_name and add_clinic_name not in st.session_state.clinics:
            add_services = [s.strip() for s in add_services_text.split("\n") if s.strip()]
            st.session_state.clinics[add_clinic_name] = {
                "clinic_type": add_clinic_type,
                "services": add_services
            }
            st.session_state.selected_clinic = add_clinic_name
            st.success(f"Added {add_clinic_name}!")
            st.rerun()
        elif add_clinic_name in st.session_state.clinics:
            st.error("Clinic already exists!")
        else:
            st.error("Please enter a clinic name")

# Delete clinic
if len(st.session_state.clinics) > 1:
    with st.sidebar.expander("🗑️ Delete Clinic", expanded=False):
        if st.button(f"Delete '{selected_clinic}'", type="secondary"):
            del st.session_state.clinics[selected_clinic]
            st.session_state.selected_clinic = list(st.session_state.clinics.keys())[0]
            st.rerun()

st.sidebar.markdown("---")

# Message source
source = st.sidebar.selectbox(
    "Message Source",
    ["instagram", "whatsapp", "facebook", "tiktok", "email"]
)

# Show current clinic info
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 {selected_clinic}")
st.sidebar.markdown(f"*{current_clinic['clinic_type']}*")
st.sidebar.markdown("**Services:**")
for svc in current_clinic["services"]:
    st.sidebar.markdown(f"- {svc}")

# Get values for classification
clinic_name = selected_clinic
clinic_type = current_clinic["clinic_type"]
services = current_clinic["services"]

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0

# Main area
st.markdown("---")

# Quick examples row
st.markdown("### 💬 Quick Test Examples")
ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

with ex_col1:
    if st.button("🎯 Lead", use_container_width=True):
        st.session_state.messages = [
            {"role": "customer", "content": "Hi, I saw your clinic on Instagram"},
            {"role": "business", "content": "Hello! How can we help you today?"},
            {"role": "customer", "content": "I'm interested in teeth whitening. How much does it cost and when can I come in?"}
        ]
        st.session_state.last_result = None

with ex_col2:
    if st.button("🚫 Spam", use_container_width=True):
        st.session_state.messages = [
            {"role": "customer", "content": "CONGRATULATIONS! You've won $1,000,000! Click here: bit.ly/scam"}
        ]
        st.session_state.last_result = None

with ex_col3:
    if st.button("❓ Unclear", use_container_width=True):
        st.session_state.messages = [
            {"role": "customer", "content": "Hi"}
        ]
        st.session_state.last_result = None

with ex_col4:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None

st.markdown("---")

# Two column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 💬 Conversation")

    # Message input form
    with st.container(border=True):
        role = st.radio("Role", ["customer", "business"], horizontal=True)
        content = st.text_area("Message", placeholder="Type a message...", height=80, label_visibility="collapsed")

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("➕ Add Message", use_container_width=True):
                if content:
                    st.session_state.messages.append({"role": role, "content": content})
                    st.session_state.last_result = None
                    st.rerun()
        with btn_col2:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                if st.session_state.messages:
                    with st.spinner("Classifying..."):
                        try:
                            conversation = ConversationInput(
                                conversation_id="test_001",
                                source=source,
                                messages=st.session_state.messages,
                                clinic_name=clinic_name,
                                clinic_type=clinic_type,
                                services=services
                            )
                            start_time = time.time()
                            st.session_state.last_result = classifier.classify(conversation)
                            st.session_state.last_response_time = time.time() - start_time
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Add at least one message")

    # Display messages
    if st.session_state.messages:
        st.markdown(f"**{len(st.session_state.messages)} message(s)** · *{source}*")

        for i, msg in enumerate(st.session_state.messages):
            is_customer = msg["role"] == "customer"

            with st.container(border=True):
                msg_col1, msg_col2 = st.columns([10, 1])
                with msg_col1:
                    if is_customer:
                        st.markdown(f"**👤 Customer**")
                    else:
                        st.markdown(f"**🏥 Business**")
                    st.markdown(msg["content"])
                with msg_col2:
                    if st.button("✕", key=f"del_{i}", help="Remove"):
                        st.session_state.messages.pop(i)
                        st.session_state.last_result = None
                        st.rerun()
    else:
        st.info("👆 Add messages above or use quick examples")

with col2:
    st.markdown("### 🎯 Classification Result")

    if st.session_state.last_result:
        result = st.session_state.last_result

        # Result card
        with st.container(border=True):
            if result.is_lead:
                st.markdown("## 🎯 LEAD")
                st.success("This conversation shows lead potential!")
            elif result.classification == "not_lead":
                st.markdown("## 🚫 NOT A LEAD")
                st.error("This conversation is not a lead.")
            else:
                st.markdown("## ❓ NEEDS INFO")
                st.warning("More context needed to classify.")

            st.markdown("---")

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Confidence", f"{result.confidence:.0%}")
            with m2:
                st.metric("Classification", result.classification.replace("_", " ").upper())
            with m3:
                response_time = st.session_state.get("last_response_time", 0)
                st.metric("Response Time", f"{response_time:.2f}s")

        # Reasoning
        with st.container(border=True):
            st.markdown("**💭 AI Reasoning**")
            st.markdown(result.reasoning)

        # Key Signals
        if result.key_signals:
            with st.container(border=True):
                st.markdown("**🔍 Key Signals**")
                for signal in result.key_signals:
                    st.markdown(f"• {signal}")

        # Details
        with st.expander("📋 Full Details"):
            st.json({
                "classification": result.classification,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "key_signals": result.key_signals,
                "is_lead": result.is_lead,
                "clinic": clinic_name,
                "source": source,
                "message_count": len(st.session_state.messages)
            })
    else:
        with st.container(border=True):
            st.markdown("#### No classification yet")
            st.markdown("Add messages and click **Classify** to see results.")

            st.markdown("---")

            st.markdown("**Current Clinic:**")
            st.markdown(f"🏥 {clinic_name}")
            st.markdown(f"*{clinic_type}*")
