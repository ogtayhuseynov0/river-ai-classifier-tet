# AI Conversation Classifier - Implementation TODO

## Phase 1: MVP (Core Classification)

### Project Setup
- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up FastAPI application structure
- [ ] Create `requirements.txt` with dependencies:
  - fastapi
  - uvicorn
  - pydantic
  - google-cloud-aiplatform
  - supabase-py
  - python-dotenv
- [ ] Create `.env.example` with required environment variables
- [ ] Set up basic logging configuration

### Health Endpoints
- [ ] Create `/health` liveness endpoint
- [ ] Create `/health/ready` readiness endpoint
- [ ] Test health endpoints locally

### Vertex AI Integration
- [ ] Set up Google Cloud credentials
- [ ] Create `vertex_ai.py` client wrapper
- [ ] Implement conversation classification prompt
- [ ] Add rate limiting with token bucket
- [ ] Add retry logic with exponential backoff
- [ ] Test Vertex AI calls locally

### Supabase Integration
- [ ] Create `supabase_client.py`
- [ ] Implement fetch conversation history
- [ ] Implement update conversation classification
- [ ] Test Supabase operations locally

### Classification Endpoint
- [ ] Create `/api/v1/classify` POST endpoint
- [ ] Accept conversation_id, return classification
- [ ] Integrate Vertex AI and Supabase clients
- [ ] Add request validation with Pydantic models
- [ ] Test classification endpoint locally

### Initial Deployment
- [ ] Create `Dockerfile` for API service
- [ ] Create `cloudbuild.yaml` for Cloud Build
- [ ] Deploy to Cloud Run (manual trigger)
- [ ] Test deployed endpoint

---

## Phase 2: Redis + Queue Integration

### Cloud Memorystore Setup
- [ ] Create Cloud Memorystore Redis instance (Basic tier, 1GB)
- [ ] Configure VPC connector for Cloud Run
- [ ] Test Redis connection from Cloud Run

### Redis Client
- [ ] Create `redis_client.py` with connection pooling
- [ ] Add health check for Redis connection
- [ ] Test Redis operations

### Conversation Cache
- [ ] Create `conversation_cache.py`
- [ ] Implement get conversation context
- [ ] Implement cache conversation context (TTL: 1 hour)
- [ ] Implement update cache with new message
- [ ] Add cache hit/miss metrics

### BullMQ Queue
- [ ] Add `bullmq` or `arq` dependency
- [ ] Create queue configuration
- [ ] Create job schema (conversation_id, triggered_at)
- [ ] Implement job debouncing (2-second delay)

### Webhook Endpoint
- [ ] Create `/api/v1/webhooks/supabase` endpoint
- [ ] Parse Supabase webhook payload
- [ ] Add job to classification queue
- [ ] Return 202 Accepted
- [ ] Add webhook signature validation

### Worker Service
- [ ] Create `queue_worker.py`
- [ ] Implement job processor:
  1. Check cache for context
  2. Fetch from Supabase if cache miss
  3. Call Vertex AI classifier
  4. Update Supabase with result
  5. Update cache
- [ ] Add error handling and dead letter queue
- [ ] Create `Dockerfile.worker`

### Deploy Queue Infrastructure
- [ ] Deploy worker service to Cloud Run
- [ ] Configure Supabase webhook to call API
- [ ] Test end-to-end flow

---

## Phase 3: Production Hardening

### Rate Limiting
- [ ] Implement token bucket rate limiter for Vertex AI
- [ ] Add queue backpressure when approaching quota
- [ ] Add metrics for rate limit hits

### Monitoring
- [ ] Add Prometheus metrics endpoint
- [ ] Create Cloud Monitoring dashboard:
  - Messages processed per hour
  - Classification distribution (lead/spam/info)
  - Average latency
  - Cache hit rate
  - Error rate
- [ ] Set up alerts:
  - Error rate > 5%
  - Queue depth > 1000
  - Vertex AI latency p95 > 2s

### Logging
- [ ] Implement structured JSON logging
- [ ] Add request ID tracing
- [ ] Configure Cloud Logging export
- [ ] Create log-based metrics

### Prompt Versioning
- [ ] Create `ai_prompts` table in Supabase
- [ ] Implement prompt version loading
- [ ] Add A/B testing support (traffic percentage)
- [ ] Create admin endpoint to update prompts

### Security Hardening
- [ ] Implement API key validation middleware
- [ ] Add request rate limiting per client
- [ ] Audit secrets access
- [ ] Enable Cloud Armor WAF (if needed)

---

## Phase 4: Future Features

### Auto-Reply Suggestions
- [ ] Design reply suggestion schema
- [ ] Create reply generator service
- [ ] Integrate with classification pipeline
- [ ] Store suggestions in Supabase

### Data Extraction
- [ ] Design entity extraction prompt
- [ ] Extract: name, phone, email, intent
- [ ] Store extracted data
- [ ] Create extraction history table

### CRM Integration
- [ ] Design CRM connector interface
- [ ] Implement HubSpot connector
- [ ] Add webhook for CRM sync
- [ ] Create sync status tracking

### Analytics Pipeline
- [ ] Export events to Pub/Sub
- [ ] Create Dataflow pipeline to BigQuery
- [ ] Build Looker Studio dashboards
- [ ] Add daily/weekly report generation

---

## Database Migrations

### Phase 1
```sql
-- Add classification columns to conversations
ALTER TABLE conversations ADD COLUMN ai_classification_type TEXT;
ALTER TABLE conversations ADD COLUMN ai_classification_confidence DECIMAL(3,2);
ALTER TABLE conversations ADD COLUMN ai_classified_at TIMESTAMPTZ;
ALTER TABLE conversations ADD COLUMN ai_classification_reasoning TEXT;
ALTER TABLE conversations ADD COLUMN ai_last_message_count INTEGER;
```

### Phase 2
```sql
-- Create classification history table
CREATE TABLE ai_classification_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  classification_type TEXT NOT NULL,
  confidence DECIMAL(3,2) NOT NULL,
  reasoning TEXT,
  message_count INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  latency_ms INTEGER,
  tokens_used INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_classification_history_conversation
  ON ai_classification_history(conversation_id);
```

### Phase 3
```sql
-- Create prompts table for versioning
CREATE TABLE ai_prompts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  prompt_type TEXT NOT NULL,
  version TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  user_prompt_template TEXT NOT NULL,
  is_active BOOLEAN DEFAULT FALSE,
  traffic_percentage INTEGER DEFAULT 100,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create feedback table for training
CREATE TABLE ai_classification_feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  classification_id UUID NOT NULL REFERENCES ai_classification_history(id),
  original_classification TEXT NOT NULL,
  corrected_classification TEXT NOT NULL,
  corrected_by UUID,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Environment Variables

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=xxx

# Redis
REDIS_HOST=10.x.x.x
REDIS_PORT=6379
REDIS_PASSWORD=xxx

# Vertex AI
GCP_PROJECT_ID=xxx
GCP_LOCATION=us-central1
VERTEX_AI_MODEL=gemini-1.5-flash

# API
API_KEY=xxx
API_HOST=0.0.0.0
API_PORT=8080

# Worker
WORKER_CONCURRENCY=10
JOB_DEBOUNCE_SECONDS=2
```

---

## Testing Checklist

### Unit Tests
- [ ] Vertex AI client mock tests
- [ ] Supabase client mock tests
- [ ] Classification logic tests
- [ ] Cache operations tests
- [ ] Queue job processing tests

### Integration Tests
- [ ] End-to-end classification flow
- [ ] Webhook to queue to classification
- [ ] Cache hit/miss scenarios
- [ ] Error handling scenarios

### Load Tests
- [ ] 100 concurrent classifications
- [ ] 1000 messages in queue
- [ ] Cache performance under load
