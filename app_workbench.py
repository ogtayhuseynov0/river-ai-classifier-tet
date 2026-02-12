"""
Prompt Engineering Workbench
Edit prompts, toggle variables, adjust model config, re-run classification/response, save presets.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Any, Dict, Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from supabase import create_client
from google import genai
from google.genai import types as genai_types

from main import EXTRACTION_PROMPT
from response_generator import RESPONSE_PROMPT, BRAND_DNA_PROFILES, format_brand_dna_prompt


# =============================================================================
# Copied from RiverAI backend - google_model_utils.py (slimmed for workbench)
# =============================================================================

class GoogleModelTypes:
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


def _extract_json_from_text(text: Optional[str]) -> str:
    """Extract JSON object from model text that may contain preamble/fences."""
    if not text:
        return "{}"
    t = text.strip()
    # Remove markdown fences
    if "```" in t:
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    # Extract first { ... last }
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


class GoogleModel:
    """Slim wrapper around google.genai for workbench use."""

    def __init__(self, project_id: str = "", location: str = "us-central1"):
        if not project_id:
            project_id = os.getenv("GOOGLE_PROJECT_ID", "").strip()
        if not location:
            location = os.getenv("GOOGLE_LOCATION", "us-central1")
        if not project_id:
            raise ValueError("Set GOOGLE_PROJECT_ID env var or pass project_id.")
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def generate_text(
        self,
        prompt: str,
        model_name: str = GoogleModelTypes.GEMINI_2_5_FLASH,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[str]:
        cfg: Dict[str, Any] = {}
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = max_output_tokens
        final_cfg = genai_types.GenerateContentConfig(**cfg) if cfg else None
        try:
            resp = self.client.models.generate_content(
                model=model_name, contents=prompt, config=final_cfg,
            )
            tokens = getattr(resp.usage_metadata, "total_token_count", 0) or 0
            return resp.text, tokens
        except Exception:
            self.logger.exception("generate_text failed.")
            return None, 0

    def generate_structured(
        self,
        prompt: str,
        response_schema: type,
        model_name: str = GoogleModelTypes.GEMINI_2_5_FLASH,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        schema_text = json.dumps(response_schema.model_json_schema(), indent=2)
        json_instruction = (
            "\n\n---\nRESPOND WITH ONLY A RAW JSON OBJECT. "
            "No markdown, no fences, no preamble, no explanation.\n"
            f"Required schema:\n{schema_text}"
        )
        full_prompt = prompt + json_instruction

        cfg: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if temperature is not None:
            cfg["temperature"] = temperature
        # Thinking models (2.5+) use thinking tokens from this budget.
        # Need headroom: 1k-2k for thinking + actual JSON output.
        tokens = max(max_output_tokens or 4096, 4096)
        cfg["max_output_tokens"] = tokens

        final_cfg = genai_types.GenerateContentConfig(**cfg)
        resp = self.client.models.generate_content(
            model=model_name, contents=full_prompt, config=final_cfg,
        )
        raw = _extract_json_from_text(resp.text)
        tokens = getattr(resp.usage_metadata, "total_token_count", 0) or 0
        return response_schema.model_validate_json(raw), tokens


# =============================================================================
# Copied from RiverAI backend - lead_classifier schemas
# =============================================================================

class MatchedServiceResponse(BaseModel):
    service: str = Field(..., description="Service name matching clinic offerings")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Match confidence")


class ExtractedDataResponse(BaseModel):
    first_name: Optional[str] = Field(None, description="Customer's first name")
    last_name: Optional[str] = Field(None, description="Customer's last name / surname")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    gender: Optional[str] = Field(None, description="Gender if mentioned (male/female/other)")
    street: Optional[str] = Field(None, description="Street name")
    house_number: Optional[str] = Field(None, description="House/building number")
    flat_number: Optional[str] = Field(None, description="Flat/apartment number")
    post_code: Optional[str] = Field(None, description="Postal/zip code")
    city: Optional[str] = Field(None, description="City name")
    country: Optional[str] = Field(None, description="Country name or ISO code")
    language: Optional[str] = Field(None, description="ISO 639-1 language code (en, es, de, etc.)")
    matched_services: List[MatchedServiceResponse] = Field(default_factory=list, description="Services customer asked about")


class ClassificationResponse(BaseModel):
    classification: Literal["lead", "not_lead", "needs_info"] = Field(..., description="Classification result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="One sentence explanation")
    key_signals: List[str] = Field(default_factory=list, description="Key signals identified")
    extracted: ExtractedDataResponse = Field(default_factory=ExtractedDataResponse, description="Extracted data")


# =============================================================================
# Copied from RiverAI backend - response_drafter schema
# =============================================================================

class DraftResponse(BaseModel):
    draft_content: str = Field(..., description="The draft response message ready to send.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Brief explanation of the response approach taken.")


# =============================================================================
# Copied from RiverAI backend - lead_classifier prompts
# =============================================================================

CLASSIFICATION_PROMPT = """Analyze this conversation for {clinic_type} "{clinic_name}".
{location_context}
Services offered:
{formatted_services}

Conversation ({source}):
{formatted_messages}

CLASSIFICATION RULES:
- lead: asking about services above, pricing, appointments, or related health concerns
- not_lead: spam, wrong number, job seekers, existing patient follow-up, or services we don't offer
- needs_info: too short/unclear to determine

GEOGRAPHIC CONSIDERATION:
- If clinic mentions they don't operate in customer's region -> classify as not_lead with high confidence
- If customer location is very far from clinic location -> reduce confidence (but don't auto-reject)
- If customer explicitly mentions being in a different country/distant city -> lower confidence by 0.1-0.2

CONFIDENCE GUIDE:
- 0.95+: explicit intent ("I want to book X")
- 0.80-0.94: strong signals (asking prices, availability)
- 0.60-0.79: moderate signals (general interest)
- 0.40-0.59: weak signals (vague message)
- <0.40: very uncertain

EXTRACTION RULES:
- Name: Split full names into first_name and last_name separately. If only one name given, put it in first_name and leave last_name empty.
- Address: Extract street, house_number, flat_number, city, post_code, country if mentioned. Extract each component separately.
- Language: Extract if mentioned (ISO code like 'es', 'en', 'fr')
- matched_services: ONLY include services the customer explicitly mentioned or asked about - do NOT list all services, leave empty if none specifically mentioned."""

DRAFT_RESPONSE_PROMPT = """You are responding on behalf of "{clinic_name}".

## ABOUT THE CLINIC
{clinic_info}

## SERVICES OFFERED
{formatted_services}

## CONVERSATION HISTORY
{formatted_messages}
{contact_context}

## RESPONSE GUIDELINES

**Voice & Tone:**
- Professional yet warm and approachable
- Empathetic when addressing concerns or questions
- Confident but not pushy about services
- Helpful and solution-oriented

**Language:**
- Respond in the same language the customer is using
- If uncertain, use: {default_language}
- Keep sentences clear and easy to understand
- Avoid medical jargon unless the customer uses it

**Content Rules:**
- Address the customer's question/concern directly
- If they ask about a specific service, provide relevant details from the services list
- Include pricing ONLY if the service shows price info
- If booking-related, mention they can book online or contact us
- If consultation required, mention the consultation requirement
- Don't make promises about availability or specific times
- Don't diagnose or give medical advice

**Format:**
- Keep response concise (1-3 sentences for simple queries)
- Use longer responses only for detailed service inquiries
- Don't use emojis unless the customer uses them
- Don't start with "Dear" - use the customer's name if known, otherwise start directly

**Contact Info (include when relevant):**
{contact_info}

Generate a response ready for clinic staff to review and send."""


def cls_format_messages(messages: list[dict]) -> str:
    """Format conversation messages for the prompt."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        content = content.replace("\n", " ").replace("\r", " ").replace('"', "'")
        content = content[:500]
        timestamp = msg.get("timestamp", "")
        if timestamp:
            formatted.append(f"[{timestamp}] {role}: {content}")
        else:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def cls_format_services(services: list[str]) -> str:
    """Format services list for the prompt."""
    return "\n".join(f"- {service}" for service in services)

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# =============================================================================
# Password Protection
# =============================================================================

def check_password():
    app_password = os.getenv("APP_PASSWORD")
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
    if st.session_state.lockout_until:
        if datetime.now() < st.session_state.lockout_until:
            remaining = (st.session_state.lockout_until - datetime.now()).seconds
            st.error(f"Too many attempts. Try again in {remaining} seconds.")
            return False
        else:
            st.session_state.login_attempts = 0
            st.session_state.lockout_until = None
    st.title("Login Required")
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

st.set_page_config(
    page_title="Prompt Workbench",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Database Connections
# =============================================================================

@st.cache_resource
def get_crm_db():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None

@st.cache_resource
def get_results_db():
    url = os.getenv("RESULTS_SUPABASE_URL")
    key = os.getenv("RESULTS_SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None

crm_db = get_crm_db()
results_db = get_results_db()

# =============================================================================
# AI Execution Engine (google.genai SDK)
# =============================================================================

GLOBAL_MODELS = {"gemini-3-pro-preview", "gemini-3-flash-preview"}

@st.cache_resource
def get_google_model(location: str = "us-central1", _v: int = 3):
    """Initialise GoogleModel once per location."""
    return GoogleModel(location=location)

def _model_location(model_name: str) -> str:
    """Gemini-3 preview models require the global endpoint."""
    return "global" if model_name in GLOBAL_MODELS else "us-central1"


def execute_structured_prompt(model_name, prompt_text, response_schema, temperature, max_tokens):
    """Execute a structured prompt via google.genai. Returns (parsed_model, raw_text, latency_ms, token_count)."""
    model = get_google_model(_model_location(model_name))
    start = time.time()
    parsed, token_count = model.generate_structured(
        prompt=prompt_text,
        response_schema=response_schema,
        model_name=model_name,
        temperature=float(temperature),
        max_output_tokens=int(max_tokens),
    )
    latency_ms = int((time.time() - start) * 1000)

    if parsed is None:
        raise Exception("generate_structured returned None - check logs")

    raw_text = parsed.model_dump_json(indent=2)
    return parsed, raw_text, latency_ms, token_count


def execute_text_prompt(model_name, prompt_text, temperature, max_tokens):
    """Execute a plain text prompt via google.genai. Returns (raw_text, latency_ms, token_count)."""
    model = get_google_model(_model_location(model_name))
    start = time.time()
    raw_text, token_count = model.generate_text(
        prompt=prompt_text,
        model_name=model_name,
        temperature=float(temperature),
        max_output_tokens=int(max_tokens),
    )
    latency_ms = int((time.time() - start) * 1000)

    if raw_text is None:
        raise Exception("generate_text returned None - check logs")

    return raw_text, latency_ms, token_count

# =============================================================================
# Data Loading (CRM DB)
# =============================================================================

@st.cache_data(ttl=300)
def load_organisations():
    if not crm_db:
        return []
    try:
        response = crm_db.table("organisations").select("id, name").order("name").execute()
        return response.data or []
    except Exception as e:
        st.error(f"Error loading organisations: {e}")
        return []

@st.cache_data(ttl=300)
def load_org_services(org_id: str):
    if not crm_db:
        return []
    try:
        response = crm_db.table("organisation_services").select(
            "id, name, description, category_id, price, currency"
        ).eq("org_id", org_id).execute()
        return response.data or []
    except Exception as e:
        st.error(f"Error loading services: {e}")
        return []

@st.cache_data(ttl=60)
def load_chats(org_id: str, channel_type: str = None, limit: int = 50):
    if not crm_db or not org_id:
        return []
    try:
        query = crm_db.schema("crm").table("chats").select(
            "id, org_id, channel_type, title, last_message_at, last_message_preview, is_archived, is_group"
        ).eq("org_id", org_id).eq("is_archived", False).eq("is_group", False).order(
            "last_message_at", desc=True
        ).limit(limit)
        if channel_type:
            query = query.ilike("channel_type", channel_type)
        return query.execute().data or []
    except Exception as e:
        st.error(f"Error loading chats: {e}")
        return []

@st.cache_data(ttl=60)
def load_chat_messages(thread_id: str, limit: int = 50):
    if not crm_db or not thread_id:
        return []
    try:
        response = crm_db.schema("crm").table("chat_messages").select(
            "id, direction, body, sent_at, received_at, created_at, contact_id"
        ).eq("thread_id", thread_id).order("created_at", desc=False).limit(limit).execute()
        return response.data or []
    except Exception as e:
        st.error(f"Error loading messages: {e}")
        return []

@st.cache_data(ttl=60)
def load_classification_logs(thread_id: str):
    if not crm_db or not thread_id:
        return {}
    try:
        response = crm_db.schema("crm").table("classification_logs").select("*").eq("thread_id", thread_id).execute()
        return {log["message_id"]: log for log in (response.data or []) if log.get("message_id")}
    except Exception as e:
        st.error(f"Error loading classification logs: {e}")
        return {}

@st.cache_data(ttl=60)
def load_draft_messages(thread_id: str):
    if not crm_db or not thread_id:
        return {}
    try:
        response = crm_db.schema("crm").table("draft_messages").select("*").eq("thread_id", thread_id).execute()
        return {d["message_id"]: d for d in (response.data or []) if d.get("message_id")}
    except Exception as e:
        st.error(f"Error loading drafts: {e}")
        return {}

# =============================================================================
# Preset CRUD (Results DB)
# =============================================================================

def load_presets(prompt_type: str = None):
    if not results_db:
        return []
    try:
        query = results_db.table("prompt_presets").select("*").order("updated_at", desc=True)
        if prompt_type:
            query = query.eq("prompt_type", prompt_type)
        return query.execute().data or []
    except:
        return []

def save_preset(name, description, tags, prompt_type, prompt_template, variables_config,
                model_name, temperature, max_output_tokens, response_mime_type, brand_dna_key):
    if not results_db:
        st.warning("Results DB not configured")
        return None
    data = {
        "name": name,
        "description": description,
        "tags": tags,
        "prompt_type": prompt_type,
        "prompt_template": prompt_template,
        "variables_config": variables_config,
        "model_name": model_name,
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "response_mime_type": response_mime_type,
        "brand_dna_key": brand_dna_key,
    }
    try:
        resp = results_db.table("prompt_presets").upsert(data, on_conflict="name").execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        st.error(f"Error saving preset: {e}")
        return None

def delete_preset(preset_id):
    if not results_db:
        return False
    try:
        results_db.table("prompt_presets").delete().eq("id", preset_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting preset: {e}")
        return False

# =============================================================================
# Test Run Persistence (Results DB)
# =============================================================================

def save_test_run(data: dict):
    if not results_db:
        return None
    try:
        resp = results_db.table("test_runs").insert(data).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        st.error(f"Error saving test run: {e}")
        return None

def load_test_runs_for_chat(thread_id: str, limit: int = 20):
    if not results_db or not thread_id:
        return []
    try:
        return results_db.table("test_runs").select("*").eq(
            "thread_id", thread_id
        ).order("created_at", desc=True).limit(limit).execute().data or []
    except:
        return []

def update_test_run_rating(run_id: str, rating: int, note: str = None):
    if not results_db:
        return False
    try:
        data = {"rating": rating}
        if note is not None:
            data["note"] = note
        results_db.table("test_runs").update(data).eq("id", run_id).execute()
        return True
    except:
        return False

# =============================================================================
# Prompt Helpers
# =============================================================================

DEFAULT_TEMPLATES = {
    "classification": CLASSIFICATION_PROMPT,
    "extraction": EXTRACTION_PROMPT,
    "response": RESPONSE_PROMPT,
}

VARIABLE_DEFS = {
    "classification": {
        "clinic_name": "Organisation name",
        "clinic_type": "Organisation type (e.g. dental clinic)",
        "location_context": "Clinic location (city, country)",
        "formatted_services": "List of services offered",
        "source": "Channel source (instagram, whatsapp, etc.)",
        "formatted_messages": "Formatted conversation messages",
    },
    "extraction": {
        "formatted_services": "List of services offered",
        "formatted_messages": "Formatted conversation messages",
    },
    "response": {
        "business_name": "Organisation name",
        "business_type": "Organisation type",
        "services_list": "List of services offered",
        "conversation_history": "Formatted conversation messages",
        "brand_dna_section": "Brand voice/style instructions",
        "channel": "Channel type for tone adjustment",
    },
}


def build_variable_values(org, services, messages, channel, brand_dna_key=None):
    """Build the actual variable values from loaded data."""
    org_name = org.get("name", "Unknown") if org else "Unknown"
    org_type = "medical practice"
    service_names = [s["name"] for s in services] if services else []

    formatted_msgs = []
    for msg in (messages or []):
        if msg.get("body"):
            formatted_msgs.append({
                "role": "customer" if msg.get("direction") == "INBOUND" else "business",
                "content": msg.get("body", ""),
            })

    brand_dna = BRAND_DNA_PROFILES.get(brand_dna_key) if brand_dna_key else None

    return {
        # Classification vars (uses backend prompt helpers)
        "clinic_name": org_name,
        "clinic_type": org_type,
        "formatted_services": cls_format_services(service_names),
        "source": channel or "chat",
        "formatted_messages": cls_format_messages(formatted_msgs),
        "location_context": "",
        # Response vars (legacy prompts from response_generator.py)
        "business_name": org_name,
        "business_type": org_type,
        "services_list": "\n".join(f"- {s}" for s in service_names[:15]),
        "conversation_history": "\n".join(
            f"{m.get('role', 'customer').upper()}: {m.get('content', '')[:500]}"
            for m in formatted_msgs[-10:]
        ),
        "brand_dna_section": format_brand_dna_prompt(brand_dna) if brand_dna else "",
        "channel": channel or "chat",
    }


def render_prompt(template: str, variables: dict, enabled_vars: dict) -> str:
    """Render a prompt template with only enabled variables filled in."""
    values = {}
    for var_name, var_value in variables.items():
        if enabled_vars.get(var_name, True):
            values[var_name] = var_value
        else:
            values[var_name] = f"[{var_name} disabled]"

    try:
        return template.format(**values)
    except KeyError as e:
        return f"[Template error: missing variable {e}]\n\n{template}"


# =============================================================================
# Trigger point callbacks
# =============================================================================

def _set_trigger(msg):
    st.session_state.wb_selected_msg = msg
    st.session_state.wb_run_result = None

def _clear_trigger():
    st.session_state.wb_selected_msg = None
    st.session_state.wb_run_result = None

# =============================================================================
# Main App
# =============================================================================

def main():
    st.title("Prompt Engineering Workbench")

    if not crm_db:
        st.error("CRM database not configured. Set SUPABASE_URL and SUPABASE_KEY.")
        return
    if not results_db:
        st.warning("Results DB not configured. Presets and test runs will not persist.")

    # =========================================================================
    # Sidebar
    # =========================================================================
    with st.sidebar:
        st.header("Data")

        # Org selector
        orgs = load_organisations()
        org_options = {org["name"]: org for org in orgs}
        if not org_options:
            st.warning("No organisations found")
            return
        selected_org_name = st.selectbox("Organisation", list(org_options.keys()))
        selected_org = org_options[selected_org_name]
        selected_org_id = selected_org["id"]

        # Services
        services = load_org_services(selected_org_id)
        service_names = [s["name"] for s in services] if services else []
        with st.expander(f"Services ({len(service_names)})"):
            for svc in service_names[:15]:
                st.markdown(f"- {svc}")
            if len(service_names) > 15:
                st.caption(f"...and {len(service_names) - 15} more")

        st.divider()

        # Channel filter
        channel_opts = ["All", "Instagram", "WhatsApp", "Facebook", "Email", "SMS", "Telegram"]
        selected_channel = st.selectbox("Channel", channel_opts)
        channel_filter = None if selected_channel == "All" else selected_channel

        # Reset pagination on filter change
        fkey = f"{selected_org_id}_{selected_channel}"
        if st.session_state.get("wb_filter_key") != fkey:
            st.session_state.wb_filter_key = fkey
            st.session_state.wb_chat_limit = 30

        if "wb_chat_limit" not in st.session_state:
            st.session_state.wb_chat_limit = 30

        # Chats
        chats = load_chats(selected_org_id, channel_type=channel_filter, limit=st.session_state.wb_chat_limit)
        st.subheader(f"Chats ({len(chats)})")

        for chat in chats:
            chat_id = chat["id"]
            title = chat.get("title") or (chat.get("last_message_preview") or "")[:20] or chat_id[:8]
            ch_icon = {"Instagram": "I", "WhatsApp": "W", "Facebook": "F", "Email": "E", "SMS": "S", "Telegram": "T"}.get(
                chat.get("channel_type", ""), ""
            )
            label = f"{ch_icon} {title[:20]}"
            if st.button(label, key=f"wb_chat_{chat_id}", use_container_width=True):
                st.session_state.wb_selected_chat = chat
                st.session_state.wb_selected_msg = None
                st.session_state.wb_msg_display_limit = 30
                st.session_state.wb_run_result = None
                st.rerun()

        if len(chats) >= st.session_state.wb_chat_limit:
            if st.button("Load more chats...", key="wb_more_chats"):
                st.session_state.wb_chat_limit += 30
                st.rerun()

        # (Messages shown in main area conversation accordion)

    # =========================================================================
    # Main Area
    # =========================================================================
    selected_chat = st.session_state.get("wb_selected_chat")

    if not selected_chat:
        st.info("Select a chat from the sidebar to start")
        return

    thread_id = selected_chat["id"]
    channel = (selected_chat.get("channel_type") or "chat").lower()
    all_messages = load_chat_messages(thread_id)
    cls_logs = load_classification_logs(thread_id)
    draft_logs = load_draft_messages(thread_id)

    # ----- Conversation accordion with trigger point selection -----
    chat_title = selected_chat.get("title") or thread_id[:12]
    selected_msg = st.session_state.get("wb_selected_msg")

    # Message count limit for loading
    if "wb_msg_display_limit" not in st.session_state:
        st.session_state.wb_msg_display_limit = 30

    # Header: show selected trigger point info
    if selected_msg:
        # Find index of selected message to compute context size
        sel_idx = next((i for i, m in enumerate(all_messages) if m.get("id") == selected_msg.get("id")), None)
        ctx_count = (sel_idx + 1) if sel_idx is not None else len(all_messages)
        sel_body = (selected_msg.get("body") or "")[:80]
        trigger_label = f"Trigger point: \"{sel_body}\" ({ctx_count}/{len(all_messages)} messages as context)"
    else:
        ctx_count = len(all_messages)
        trigger_label = f"No trigger point selected - using all {len(all_messages)} messages"

    st.markdown(f"**Chat:** {chat_title} | **Channel:** {channel} | {trigger_label}")

    # Pre-compute trigger index once
    _trigger_idx = None
    if selected_msg:
        _trigger_idx = next((i for i, m in enumerate(all_messages) if m.get("id") == selected_msg.get("id")), None)

    with st.expander("Conversation", expanded=True):
        # Scrollable container
        with st.container(height=400):
            display_limit = st.session_state.wb_msg_display_limit
            total = len(all_messages)

            start_idx = max(0, total - display_limit)
            visible_messages = all_messages[start_idx:]

            # "Load older" button at top
            if start_idx > 0:
                if st.button(f"Load older messages ({start_idx} remaining)", key="wb_load_older", use_container_width=True):
                    st.session_state.wb_msg_display_limit += 30
                    st.rerun()

            # Column headers
            h1, h2, h3 = st.columns([2, 2, 2])
            with h1:
                st.markdown("**Message**")
            with h2:
                st.markdown("**Classification**")
            with h3:
                st.markdown("**Draft Response**")
            st.divider()

            # Render each message row
            for idx, msg in enumerate(visible_messages):
                real_idx = start_idx + idx
                msg_id = msg.get("id", "")
                is_customer = msg.get("direction") == "INBOUND"
                body = msg.get("body") or ""
                created_at = msg.get("created_at", "")

                is_selected = bool(selected_msg and selected_msg.get("id") == msg_id)
                is_after_trigger = _trigger_idx is not None and real_idx > _trigger_idx

                # Parse time
                time_str = ""
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        time_str = dt.strftime("%b %d, %H:%M")
                    except:
                        time_str = created_at[:16]

                # --- 3-column row ---
                col_msg, col_cls, col_draft = st.columns([2, 2, 2])

                # Column 1: Message card + trigger button
                with col_msg:
                    if is_customer:
                        icon, lbl = "👤", "Customer"
                        bg = "#1b3a4b" if is_selected else ("#e3f2fd" if not is_after_trigger else "#f5f5f5")
                        border = "#FF9800" if is_selected else "#1976d2"
                        txt = "#ffffff" if is_selected else ("#1a1a1a" if not is_after_trigger else "#999")
                    else:
                        icon, lbl = "🏥", "Business"
                        bg = "#f5f5f5" if is_after_trigger else "#e8f5e9"
                        border = "#388e3c"
                        txt = "#999" if is_after_trigger else "#1a1a1a"

                    trigger_badge = ' <span style="background:#FF9800;color:white;padding:1px 5px;border-radius:3px;font-size:11px;">TRIGGER</span>' if is_selected else ""
                    opacity = "0.5" if is_after_trigger else "1"

                    st.markdown(f"""
                    <div style="background:{bg};padding:10px;border-radius:8px;margin:4px 0;color:{txt};border-left:3px solid {border};opacity:{opacity};">
                        <small style="color:{'#ccc' if is_selected else '#555'};">{icon} <b>{lbl}</b> · {time_str}{trigger_badge}</small><br/>
                        <span style="color:{txt};">{body[:500]}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Small inline trigger button for customer messages
                    if is_customer and not is_selected:
                        st.button("Select trigger", key=f"wb_trig_{msg_id}", on_click=_set_trigger, args=(msg,))
                    elif is_customer and is_selected:
                        st.button("Clear trigger", key=f"wb_untrig_{msg_id}", on_click=_clear_trigger)

                # Column 2: Classification
                with col_cls:
                    cls_log = cls_logs.get(msg_id)
                    if cls_log:
                        cls_type = cls_log.get("classification", "?")
                        cls_conf = cls_log.get("confidence", 0)
                        badge_colors = {"lead": ("🟢", "#4CAF50"), "not_lead": ("🔴", "#f44336"), "needs_info": ("🟡", "#FFC107")}
                        cls_icon, _ = badge_colors.get(cls_type, ("⚪", "#9E9E9E"))
                        st.markdown(f"{cls_icon} **{cls_type.upper()}** ({cls_conf:.0%})")
                        if cls_log.get("reasoning"):
                            with st.expander("Reasoning"):
                                st.write(cls_log["reasoning"])
                        if cls_log.get("key_signals"):
                            signals = cls_log["key_signals"]
                            if isinstance(signals, list):
                                st.caption(", ".join(signals))
                        st.caption(f"{cls_log.get('model_name', '')} | {cls_log.get('processing_time_ms', 0)}ms")
                    else:
                        st.markdown("—")

                # Column 3: Draft
                with col_draft:
                    draft = draft_logs.get(msg_id)
                    if draft:
                        content = draft.get("draft_content", "")
                        st.info(content[:200] + ("..." if len(content) > 200 else ""))
                        conf = draft.get("confidence", 0)
                        status = draft.get("status", "")
                        st.caption(f"Confidence: {conf:.0%} | {status}")
                        st.caption(f"{draft.get('model_name', '')} | {draft.get('processing_time_ms', 0)}ms")
                    else:
                        st.markdown("—")

                st.divider()

        # Clear trigger button outside scrollable area
        if selected_msg:
            if st.button("Clear trigger point (use full conversation)", key="wb_clear_trigger"):
                st.session_state.wb_selected_msg = None
                st.session_state.wb_run_result = None
                st.rerun()

    # Slice messages to trigger point for prompt context
    if selected_msg:
        sel_idx = next((i for i, m in enumerate(all_messages) if m.get("id") == selected_msg.get("id")), None)
        if sel_idx is not None:
            messages = all_messages[:sel_idx + 1]
        else:
            messages = all_messages
    else:
        messages = all_messages

    # ----- Prompt Editor + Result -----
    # Initialise session defaults
    if "wb_prompt_type" not in st.session_state:
        st.session_state.wb_prompt_type = "classification"

    col_editor, col_result = st.columns([1, 1], gap="large")

    with col_editor:
        st.markdown("### Prompt Editor")

        # Prompt type
        prompt_type = st.radio(
            "Type",
            ["classification", "extraction", "response"],
            horizontal=True,
            key="wb_prompt_type",
        )

        # Load default template if switching types or first load
        template_key = f"wb_template_{prompt_type}"
        if template_key not in st.session_state:
            st.session_state[template_key] = DEFAULT_TEMPLATES.get(prompt_type, "")

        # Template editor
        template = st.text_area(
            "Prompt template",
            value=st.session_state[template_key],
            height=300,
            key=f"wb_editor_{prompt_type}",
        )
        st.session_state[template_key] = template

        if st.button("Reset to default", key="wb_reset_template"):
            st.session_state[template_key] = DEFAULT_TEMPLATES.get(prompt_type, "")
            st.rerun()

        # Variable toggles
        st.markdown("**Variables**")
        var_defs = VARIABLE_DEFS.get(prompt_type, {})
        vars_config_key = f"wb_vars_{prompt_type}"
        if vars_config_key not in st.session_state:
            st.session_state[vars_config_key] = {k: True for k in var_defs}

        enabled_vars = st.session_state[vars_config_key]
        for var_name, var_desc in var_defs.items():
            enabled_vars[var_name] = st.checkbox(
                f"{var_name} - {var_desc}",
                value=enabled_vars.get(var_name, True),
                key=f"wb_var_{prompt_type}_{var_name}",
            )
        st.session_state[vars_config_key] = enabled_vars

        # Brand DNA selector (response type only)
        brand_dna_key = None
        if prompt_type == "response":
            dna_options = ["None"] + list(BRAND_DNA_PROFILES.keys())
            dna_labels = ["None"] + [p["name"] for p in BRAND_DNA_PROFILES.values()]
            dna_idx = st.selectbox("Brand DNA", dna_labels, index=0, key="wb_brand_dna")
            if dna_idx != "None":
                brand_dna_key = dna_options[dna_labels.index(dna_idx)]

        st.divider()

        # Model config
        st.markdown("**Model Config**")
        model_options = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        model_name = st.selectbox("Model", model_options, key="wb_model")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05, key="wb_temp")
        max_tokens = 4096  # internal default, not shown in UI

        # Comparison mode toggle
        comparison_mode = st.checkbox("Comparison mode (A/B)", key="wb_compare_mode")

        if comparison_mode:
            st.markdown("**Preset B config**")
            model_b = st.selectbox("Model B", model_options, key="wb_model_b")
            temp_b = st.slider("Temperature B", 0.0, 2.0, 0.1, 0.05, key="wb_temp_b")

        # Build rendered prompt preview
        var_values = build_variable_values(selected_org, services, messages, channel, brand_dna_key)
        rendered_prompt = render_prompt(template, var_values, enabled_vars)

        # Schema for structured types
        schema_map = {
            "classification": ClassificationResponse,
            "extraction": ExtractedDataResponse,
        }

        with st.expander("Rendered prompt preview"):
            st.code(rendered_prompt, language="text")

        # --- Helper to run a single prompt ---
        def _run_single(m_name, temp, m_tokens):
            schema = schema_map.get(prompt_type)
            if schema:
                parsed, raw, lat, tok = execute_structured_prompt(
                    m_name, rendered_prompt, schema, temp, m_tokens,
                )
                return raw, lat, tok
            else:
                raw, lat, tok = execute_text_prompt(
                    m_name, rendered_prompt, temp, m_tokens,
                )
                return raw, lat, tok

        # Run buttons
        if comparison_mode:
            if st.button("Run Both (A vs B)", type="primary", use_container_width=True, key="wb_run_both"):
                with st.spinner("Running A..."):
                    try:
                        raw_a, lat_a, tok_a = _run_single(model_name, temperature, max_tokens)
                    except Exception as e:
                        raw_a, lat_a, tok_a = f"ERROR: {e}", 0, 0
                with st.spinner("Running B..."):
                    try:
                        raw_b, lat_b, tok_b = _run_single(model_b, temp_b, max_tokens)
                    except Exception as e:
                        raw_b, lat_b, tok_b = f"ERROR: {e}", 0, 0
                st.session_state.wb_run_result = {
                    "comparison": True,
                    "a": {"raw": raw_a, "latency_ms": lat_a, "token_count": tok_a, "model": model_name, "temp": temperature, "max_tokens": max_tokens},
                    "b": {"raw": raw_b, "latency_ms": lat_b, "token_count": tok_b, "model": model_b, "temp": temp_b, "max_tokens": max_tokens},
                    "prompt_snapshot": rendered_prompt,
                    "prompt_type": prompt_type,
                    "vars_config": enabled_vars,
                }
                for label, r in [("a", st.session_state.wb_run_result["a"]), ("b", st.session_state.wb_run_result["b"])]:
                    _save_run_from_result(r, prompt_type, rendered_prompt, enabled_vars, thread_id, channel, brand_dna_key)
                st.rerun()
        else:
            if st.button("Run", type="primary", use_container_width=True, key="wb_run"):
                try:
                    with st.spinner("Running prompt..."):
                        raw, lat, tok = _run_single(model_name, temperature, max_tokens)
                    st.session_state.wb_run_result = {
                        "comparison": False,
                        "raw": raw,
                        "latency_ms": lat,
                        "token_count": tok,
                        "model": model_name,
                        "temp": temperature,
                        "max_tokens": max_tokens,
                        "prompt_snapshot": rendered_prompt,
                        "prompt_type": prompt_type,
                        "vars_config": enabled_vars,
                    }
                    _save_run_from_result(
                        st.session_state.wb_run_result, prompt_type, rendered_prompt,
                        enabled_vars, thread_id, channel, brand_dna_key
                    )
                    st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

    # =========================================================================
    # Result Panel
    # =========================================================================
    with col_result:
        st.markdown("### Result")

        result = st.session_state.get("wb_run_result")
        if result and result.get("comparison"):
            # Side by side comparison
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Run A**")
                _render_single_result(result["a"], result["prompt_type"], key_suffix="a")
            with cb:
                st.markdown("**Run B**")
                _render_single_result(result["b"], result["prompt_type"], key_suffix="b")
        elif result:
            _render_single_result(result, result.get("prompt_type", "classification"), key_suffix="main")
        else:
            st.info("Click Run to see results")

        # ----- Run History -----
        st.divider()
        st.markdown("### Run History")
        runs = load_test_runs_for_chat(thread_id, limit=15)
        if runs:
            for run in runs:
                run_id = run["id"]
                ts = run.get("created_at", "")[:19]
                rtype = run.get("prompt_type", "?")
                model = run.get("model_name", "?")
                cls = run.get("classification", "")
                conf = run.get("confidence")
                lat = run.get("latency_ms", 0)
                rating = run.get("rating")

                rating_icon = ""
                if rating == 1:
                    rating_icon = " [+1]"
                elif rating == -1:
                    rating_icon = " [-1]"

                if cls and conf is not None:
                    label = f"[{ts}] {rtype} | {model} | {cls.upper()} {conf:.0%} | {lat}ms{rating_icon}"
                else:
                    resp_preview = (run.get("generated_response") or run.get("raw_response") or "")[:30]
                    label = f"[{ts}] {rtype} | {model} | {lat}ms | {resp_preview}{rating_icon}"

                with st.expander(label):
                    if run.get("parsed_result"):
                        st.json(run["parsed_result"])
                    elif run.get("generated_response"):
                        st.success(run["generated_response"])
                    elif run.get("raw_response"):
                        st.code(run["raw_response"][:500])

                    st.caption(f"Tokens: {run.get('token_count', 0)} | Temp: {run.get('temperature', '?')}")

                    # Rating controls
                    rc1, rc2, rc3 = st.columns([1, 1, 3])
                    with rc1:
                        if st.button("+1", key=f"wb_up_{run_id}"):
                            update_test_run_rating(run_id, 1)
                            st.rerun()
                    with rc2:
                        if st.button("-1", key=f"wb_down_{run_id}"):
                            update_test_run_rating(run_id, -1)
                            st.rerun()
                    with rc3:
                        run_note = st.text_input("Note", value=run.get("note", "") or "", key=f"wb_rnote_{run_id}")
                        if st.button("Save", key=f"wb_rnsave_{run_id}"):
                            update_test_run_rating(run_id, run.get("rating") or 0, run_note)
                            st.rerun()
        else:
            st.caption("No runs yet for this chat")

    # =========================================================================
    # Presets Bar (bottom)
    # =========================================================================
    st.divider()
    st.markdown("### Presets")
    presets = load_presets(prompt_type)

    pcol1, pcol2 = st.columns([3, 1])
    with pcol1:
        if presets:
            preset_names = [p["name"] for p in presets]
            selected_preset_name = st.selectbox("Load preset", ["-- Select --"] + preset_names, key="wb_preset_select")
            if selected_preset_name != "-- Select --":
                preset = next(p for p in presets if p["name"] == selected_preset_name)
                if st.button("Load", key="wb_preset_load"):
                    _load_preset_into_state(preset)
                    st.rerun()
                if st.button("Delete", key="wb_preset_delete"):
                    if delete_preset(preset["id"]):
                        st.success(f"Deleted '{selected_preset_name}'")
                        st.rerun()
        else:
            st.caption("No presets saved for this prompt type")

    with pcol2:
        st.markdown("**Save current as preset**")
        preset_name = st.text_input("Preset name", key="wb_preset_name")
        preset_desc = st.text_input("Description", key="wb_preset_desc")
        preset_tags = st.text_input("Tags (comma-separated)", key="wb_preset_tags")
        if st.button("Save Preset", key="wb_preset_save", use_container_width=True):
            if not preset_name:
                st.warning("Enter a preset name")
            else:
                tags = [t.strip() for t in preset_tags.split(",") if t.strip()] if preset_tags else []
                saved = save_preset(
                    name=preset_name,
                    description=preset_desc,
                    tags=tags,
                    prompt_type=prompt_type,
                    prompt_template=template,
                    variables_config=enabled_vars,
                    model_name=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json" if prompt_type in ("classification", "extraction") else "text/plain",
                    brand_dna_key=brand_dna_key,
                )
                if saved:
                    st.success(f"Preset '{preset_name}' saved!")
                    st.rerun()


# =============================================================================
# Helpers
# =============================================================================

def _render_single_result(result_data, prompt_type, key_suffix=""):
    """Render a single run result in the result panel."""
    raw = result_data.get("raw", "")
    latency = result_data.get("latency_ms", 0)
    tokens = result_data.get("token_count", 0)
    model = result_data.get("model", "?")

    if raw.startswith("ERROR:"):
        st.error(raw)
        return

    if prompt_type in ("classification", "extraction"):
        # Try parse JSON
        try:
            parsed = json.loads(raw)
        except:
            # Try extract JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except:
                    parsed = None
            else:
                parsed = None

        if parsed and prompt_type == "classification":
            cls = parsed.get("classification", "?")
            conf = parsed.get("confidence", 0)
            reasoning = parsed.get("reasoning", "")
            signals = parsed.get("key_signals", [])

            colors = {"lead": "green", "not_lead": "red", "needs_info": "orange"}
            color = colors.get(cls.lower(), "grey")
            st.markdown(f":{color}[**{cls.upper()}**] ({conf:.0%})")

            if reasoning:
                st.markdown(f"**Reasoning:** {reasoning}")
            if signals:
                st.markdown("**Key signals:** " + ", ".join(signals))
        elif parsed:
            st.json(parsed)
        else:
            st.code(raw[:1000])
    else:
        # Response type - show as text
        st.success(raw)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Latency", f"{latency}ms")
    with m2:
        st.metric("Tokens", tokens)
    with m3:
        st.caption(f"Model: {model}")

    # Rating
    r1, r2 = st.columns(2)
    with r1:
        if st.button("+1", key=f"wb_res_up_{key_suffix}"):
            _rate_latest_run(1)
    with r2:
        if st.button("-1", key=f"wb_res_down_{key_suffix}"):
            _rate_latest_run(-1)

    note = st.text_input("Note", key=f"wb_res_note_{key_suffix}")
    if st.button("Save note", key=f"wb_res_nsave_{key_suffix}"):
        _rate_latest_run(0, note)
        st.success("Saved")


def _save_run_from_result(result_data, prompt_type, rendered_prompt, vars_config, thread_id, channel, brand_dna_key):
    """Save a test run to the DB."""
    raw = result_data.get("raw", "")

    parsed = None
    classification = None
    confidence = None
    reasoning = None
    key_signals = None
    generated_response = None

    if prompt_type in ("classification", "extraction"):
        try:
            parsed = json.loads(raw)
        except:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except:
                    pass

        if parsed and prompt_type == "classification":
            classification = parsed.get("classification")
            confidence = parsed.get("confidence")
            reasoning = parsed.get("reasoning")
            key_signals = parsed.get("key_signals")
    else:
        generated_response = raw

    selected_org = st.session_state.get("wb_selected_chat", {}).get("org_id", "")
    org_name = ""
    for org in load_organisations():
        if org["id"] == selected_org:
            org_name = org["name"]
            break

    run_data = {
        "prompt_type": prompt_type,
        "org_id": selected_org or "unknown",
        "org_name": org_name,
        "thread_id": thread_id,
        "channel": channel,
        "prompt_snapshot": rendered_prompt,
        "model_name": result_data.get("model", ""),
        "temperature": result_data.get("temp"),
        "max_output_tokens": result_data.get("max_tokens"),
        "variables_config": vars_config,
        "raw_response": raw,
        "parsed_result": parsed,
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_signals": key_signals,
        "generated_response": generated_response,
        "brand_dna_key": brand_dna_key,
        "latency_ms": result_data.get("latency_ms", 0),
        "token_count": result_data.get("token_count", 0),
    }

    saved = save_test_run(run_data)
    if saved:
        st.session_state.wb_last_run_id = saved["id"]


def _rate_latest_run(rating, note=None):
    """Rate the most recent run."""
    run_id = st.session_state.get("wb_last_run_id")
    if run_id:
        update_test_run_rating(run_id, rating, note)


def _load_preset_into_state(preset):
    """Load a preset's values into session state."""
    ptype = preset["prompt_type"]
    template_key = f"wb_template_{ptype}"
    vars_key = f"wb_vars_{ptype}"

    st.session_state[template_key] = preset["prompt_template"]
    st.session_state[vars_key] = preset.get("variables_config", {})
    st.session_state.wb_prompt_type = ptype

    # Model config is loaded but selectbox/slider state is trickier
    # We set them via session state keys
    st.session_state.wb_model = preset.get("model_name", "gemini-2.5-flash-lite")
    st.session_state.wb_temp = preset.get("temperature", 0.1)
    if isinstance(st.session_state.wb_temp, str):
        st.session_state.wb_temp = float(st.session_state.wb_temp)
    st.session_state.wb_max_tokens = preset.get("max_output_tokens", 256)

    if preset.get("brand_dna_key"):
        dna = BRAND_DNA_PROFILES.get(preset["brand_dna_key"])
        if dna:
            st.session_state.wb_brand_dna = dna["name"]


if __name__ == "__main__":
    main()
