# AI Lead Classifier

AI-powered conversation classifier for medical practices. Classifies inbox conversations (Instagram, WhatsApp, Facebook, Email) as **Lead**, **Not Lead**, or **Needs Info**.

## Features

- **Classification**: Analyzes conversations and classifies lead potential
- **Data Extraction**: Extracts customer info (name, DOB, location, etc.)
- **Service Matching**: Matches customer inquiries to clinic services with pricing
- **Batch Processing**: Generate Excel reports for multiple organizations
- **Results Storage**: Save classification results to separate database

## Tech Stack

- **AI**: Google Vertex AI (Gemini 2.0 Flash Lite)
- **Database**: Supabase (CRM data - read only)
- **Results DB**: Separate Supabase instance (write)
- **UI**: Streamlit
- **Export**: Pandas + openpyxl (Excel)

## Files

| File | Description |
|------|-------------|
| `main.py` | Core classifier - LeadClassifier, ExtractedData classes |
| `app_supabase.py` | Streamlit UI connected to Supabase CRM |
| `app.py` | Simple test UI (no database) |
| `batch_classify.py` | Batch Excel report generator |
| `results_table.sql` | SQL schema for results database |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your keys
```

Required environment variables:
```
# AI - Vertex AI key (starts with AQ.) or Google AI Studio key (starts with AIza)
GEMINI_API_KEY=your-key

# Source Database (CRM - READ ONLY)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Results Database (optional - for saving classification results)
RESULTS_SUPABASE_URL=https://your-results-project.supabase.co
RESULTS_SUPABASE_KEY=your-results-service-role-key

# App password (optional)
APP_PASSWORD=your-password
```

### 3. Setup Results Database (optional)
Run `results_table.sql` in your results Supabase instance.

### 4. Run

**Streamlit UI:**
```bash
streamlit run app_supabase.py
```

**Batch Processing:**
```bash
# Edit ORG_IDS in batch_classify.py first
python batch_classify.py
```

## Classification Logic

| Ground Truth | AI Result | Match Status |
|--------------|-----------|--------------|
| Lead | LEAD | Match |
| Lead | NOT_LEAD | Mismatch |
| Customer | NOT_LEAD | Match |
| Customer | LEAD | Customer (was lead) |
| Any | NEEDS_INFO | Needs Info |

## Data Extraction

Extracts from conversations:
- Name (first, last, middle)
- Date of birth, gender
- Address (street, city, country, post code)
- Language (ISO code)
- Occupation
- Matched services (with confidence)
- Metadata (phone, email, preferences)

## Database Safety

**IMPORTANT**: The main CRM database is READ-ONLY.

- `supabase` client: Only SELECT queries
- `results_db` client: INSERT/UPDATE to `classification_results` table only

## Configuration (batch_classify.py)

```python
# Organizations to process
ORG_IDS = ["org_xxx", "org_yyy"]

# Limit chats per org (None for all)
CHAT_LIMIT = 500

# Only process chats with lifecycle set (Lead/Customer)
RUN_ONLY_MARKED = True
```

## Excel Output

Columns:
- clinic_name, chat_id, channel, contact_name
- last_message_at, ground_truth
- ai_classification, confidence, match
- reasoning, key_signals, last_5_messages
- ext_first_name, ext_last_name, ext_date_of_birth, ext_gender
- ext_city, ext_country, ext_language, ext_occupation
- ext_matched_services, ext_metadata, note

Sheets:
- **Summary**: Stats per organization (accuracy, matches, mismatches)
- **[Org Name]**: Detailed results per organization
