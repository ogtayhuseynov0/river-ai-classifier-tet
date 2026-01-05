# AI Conversation Classifier - Architecture Documentation

## Project Overview

A standalone microservice that classifies medical practice **conversations/inboxes** from multiple platforms (Instagram, WhatsApp, TikTok, Facebook, Email) into:

- **Lead** - Potential patients showing interest
- **Spam** - Unsolicited/irrelevant messages
- **Info Needed** - Requires clarification

**Key Design Decision**: Classification is at the **conversation level**, not per-message. Each new message triggers re-classification of the entire conversation with growing context.

---

## Tech Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Backend | FastAPI (Python) | Matches existing platform, better Vertex AI SDK |
| Queue | Redis + BullMQ | Low latency, also serves as conversation cache |
| Cache | Cloud Memorystore (Redis) | Cache conversation context for faster AI calls |
| AI | Vertex AI Gemini 1.5 Flash | Fast, cost-effective classification |
| Hosting | Cloud Run | Auto-scaling, pay-per-use |
| Database | Supabase (existing) | Direct write-back |

---

## System Architecture

### High-Level Flow

```
Supabase (New Message)
    → Webhook to Cloud Run API
    → Redis Queue (BullMQ)
    → Worker Service
    → Check Redis Cache for conversation context
    → If cache miss: fetch from Supabase, then cache
    → Vertex AI classification
    → Write result to Supabase conversations table
    → Update Redis cache
```

### Classification Flow (Detailed)

1. **Trigger**: New message arrives in Supabase
2. **Webhook**: Supabase webhook calls Cloud Run API endpoint
3. **Queue**: API adds job to Redis queue with `conversation_id`
4. **Debounce**: Queue waits 2 seconds for additional messages (batch rapid messages)
5. **Cache Check**: Worker checks Redis for cached conversation context
6. **Context Build**:
   - Cache hit: use cached messages
   - Cache miss: fetch conversation history from Supabase, cache with 1-hour TTL
7. **Classify**: Send conversation context to Vertex AI Gemini
8. **Store**: Write classification result to `conversations` table
9. **Update Cache**: Add new classification to cached context

### Debouncing Logic

When multiple messages arrive rapidly:
- First message triggers a job with 2-second delay
- Subsequent messages within 2 seconds extend the delay
- After 2 seconds of quiet, classification runs once with all messages

---

## Project Structure

```
ai-classifier-service/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                    # FastAPI application entry
│       ├── config.py                  # Environment configuration
│       ├── dependencies.py            # Dependency injection
│       │
│       ├── api/
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── router.py          # API router aggregator
│       │       └── endpoints/
│       │           ├── __init__.py
│       │           ├── health.py      # Health check endpoints
│       │           ├── classify.py    # Manual classification endpoint
│       │           ├── webhooks.py    # Supabase webhook receiver
│       │           └── admin.py       # Admin/metrics endpoints
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── security.py            # Auth & API key validation
│       │   ├── logging.py             # Structured logging
│       │   └── exceptions.py          # Custom exceptions
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── conversation.py        # Conversation Pydantic models
│       │   ├── classification.py      # Classification result models
│       │   └── enums.py               # Classification types enum
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── classifier.py          # Main classification orchestration
│       │   ├── vertex_ai.py           # Vertex AI client wrapper
│       │   ├── supabase_client.py     # Supabase operations
│       │   ├── redis_client.py        # Redis connection setup
│       │   └── conversation_cache.py  # Conversation context caching
│       │
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── queue_worker.py        # BullMQ worker
│       │   └── job_processor.py       # Classification job processing
│       │
│       └── prompts/
│           ├── __init__.py
│           └── classification.py      # Prompt templates
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   └── integration/
│
├── scripts/
│   ├── run_worker.py                  # Worker entry point
│   └── run_api.py                     # API entry point
│
├── Dockerfile
├── Dockerfile.worker
├── docker-compose.yml
├── docker-compose.dev.yml
├── cloudbuild.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Database Schema

### Modifications to Existing Tables

Add to `conversations` (or `inboxes`) table:

```sql
-- Classification is at CONVERSATION level
ALTER TABLE conversations ADD COLUMN ai_classification_type TEXT;
-- Values: 'lead', 'spam', 'info_needed'

ALTER TABLE conversations ADD COLUMN ai_classification_confidence DECIMAL(3,2);
-- Range: 0.00 to 1.00

ALTER TABLE conversations ADD COLUMN ai_classified_at TIMESTAMPTZ;
-- When classification occurred

ALTER TABLE conversations ADD COLUMN ai_classification_reasoning TEXT;
-- AI's explanation for the classification

ALTER TABLE conversations ADD COLUMN ai_last_message_count INTEGER;
-- Number of messages when last classified (for tracking re-classifications)
```

### New Tables

```sql
-- Audit trail for all classifications
CREATE TABLE ai_classification_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

  -- Classification result
  classification_type TEXT NOT NULL,
  confidence DECIMAL(3,2) NOT NULL,
  reasoning TEXT,

  -- Context at time of classification
  message_count INTEGER NOT NULL,

  -- Model information
  model_id TEXT NOT NULL,
  model_version TEXT NOT NULL,

  -- Performance tracking
  latency_ms INTEGER,
  tokens_used INTEGER,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_classification_history_conversation
  ON ai_classification_history(conversation_id);
CREATE INDEX idx_classification_history_created
  ON ai_classification_history(created_at);


-- Human feedback for model improvement (future fine-tuning)
CREATE TABLE ai_classification_feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  classification_id UUID NOT NULL REFERENCES ai_classification_history(id),

  original_classification TEXT NOT NULL,
  corrected_classification TEXT NOT NULL,
  corrected_by UUID REFERENCES auth.users(id),
  feedback_notes TEXT,

  -- For training data export
  included_in_training BOOLEAN DEFAULT FALSE,

  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Redis Cache Structure

```
Key: conv:{conversation_id}
Type: Hash

Fields:
  messages      -> JSON array of last 50 messages
  classification -> Current classification type
  confidence    -> Current confidence score
  message_count -> Number of messages in conversation
  last_updated  -> ISO timestamp

TTL: 3600 seconds (1 hour)
```

---

## API Endpoints

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe (checks Redis, Supabase, Vertex AI) |
| `/metrics` | GET | Prometheus metrics |

### Classification

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/classify` | POST | Manual single conversation classification |
| `/api/v1/classify/batch` | POST | Batch classify multiple conversations |

### Webhooks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/webhooks/supabase` | POST | Receive Supabase database webhooks |

### Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/admin/queue/stats` | GET | Queue depth and processing stats |
| `/api/v1/admin/cache/stats` | GET | Cache hit/miss rates |

---

## Vertex AI Configuration

### Model Selection

| Scenario | Model | Latency | Cost |
|----------|-------|---------|------|
| Standard classification | gemini-1.5-flash | ~200ms | Low |
| Low confidence retry | gemini-1.5-pro | ~500ms | Medium |

### Rate Limiting

- **Quota**: 300 requests per minute
- **Burst**: 50 requests
- **Retry**: Exponential backoff (1s, 2s, 4s, 8s, 16s)
- **Fallback**: Queue messages when quota exhausted

### Prompt Template

```python
SYSTEM_PROMPT = """
You are classifying customer conversations for a healthcare practice.
Analyze the ENTIRE conversation and classify it into one category.

CLASSIFICATION TYPES:
1. LEAD - Customer interested in services, appointments, or treatments
2. SPAM - Unsolicited promotions, scams, or irrelevant messages
3. INFO_NEEDED - Unclear intent, requires clarification

Respond with JSON:
{
  "classification": "LEAD" | "SPAM" | "INFO_NEEDED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "key_indicators": ["list", "of", "indicators"]
}
"""
```

---

## Cloud Run Services

### API Service (`ai-classifier-api`)

| Setting | Value |
|---------|-------|
| Min instances | 1 |
| Max instances | 10 |
| CPU | 1 |
| Memory | 512Mi |
| Concurrency | 80 |
| Timeout | 60s |

### Worker Service (`ai-classifier-worker`)

| Setting | Value |
|---------|-------|
| Min instances | 2 |
| Max instances | 20 |
| CPU | 2 |
| Memory | 1Gi |
| Concurrency | 10 |
| Timeout | 300s |

---

## Cost Estimate (~10K conversations/day)

| Component | Monthly Cost |
|-----------|--------------|
| Cloud Run (API + Worker) | $65-130 |
| Cloud Memorystore (Redis, 1GB) | $50-100 |
| Vertex AI (Gemini Flash) | $150-300 |
| Cloud Logging/Monitoring | $20-40 |
| **Total** | **$300-600** |

---

## Security

### Authentication

- **Between services**: API key in `X-API-Key` header
- **Supabase webhook**: HMAC signature validation
- **Admin endpoints**: JWT or API key with admin scope

### Secrets Management

Store in Google Secret Manager:
- `supabase-url`
- `supabase-service-key`
- `api-key-main-platform`
- `redis-host`
- `redis-password`

### Data Privacy

- No PII in logs (only conversation IDs)
- Message content only in Redis cache (TTL-based expiry)
- Classification history stores reasoning, not raw messages

---

## Future Architecture (Designed For)

### Auto-Reply Suggestions
```
Classification Result → Reply Generator Service → Suggested Replies Table
```

### Data Extraction Pipeline
```
Conversation → Entity Extractor → Extracted Data (name, phone, intent)
```

### CRM Integration
```
Classification + Extraction → CRM Connector → External CRM (HubSpot, etc.)
```

### Analytics
```
All Events → Pub/Sub → Dataflow → BigQuery → Looker Dashboards
```
