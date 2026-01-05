# AI Conversation Classifier - Architecture Diagrams

## System Architecture

```mermaid
flowchart TB
    subgraph "Existing Platform"
        SP[(Supabase DB)]
        WH[Database Webhook]
    end

    subgraph "AI Classifier Service"
        subgraph "Cloud Run - API"
            API[FastAPI<br/>Webhook Receiver]
            CE[Classification<br/>Endpoint]
        end

        subgraph "Cloud Memorystore"
            RQ[Redis Queue<br/>BullMQ]
            RC[Redis Cache<br/>Conversation Context]
        end

        subgraph "Cloud Run - Worker"
            W[Queue Worker]
            JP[Job Processor]
            CL[Classifier Service]
        end
    end

    subgraph "Google Cloud AI"
        VA[Vertex AI<br/>Gemini 1.5 Flash]
    end

    SP -->|New Message Event| WH
    WH -->|POST /webhooks/supabase| API
    API -->|Add Job| RQ
    RQ -->|Pull Job| W
    W -->|Check Cache| RC
    RC -.->|Cache Miss| SP
    W --> JP
    JP --> CL
    CL -->|Classify| VA
    VA -->|Result| CL
    CL -->|Update Cache| RC
    CL -->|Write Result| SP
    CE -->|Manual Classify| CL

    style SP fill:#3ECF8E,color:#000
    style VA fill:#4285F4,color:#fff
    style RQ fill:#DC382D,color:#fff
    style RC fill:#DC382D,color:#fff
    style API fill:#009688,color:#fff
    style W fill:#009688,color:#fff
```

---

## Classification Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant S as Supabase
    participant A as API Service
    participant Q as Redis Queue
    participant C as Redis Cache
    participant W as Worker
    participant V as Vertex AI

    Note over S: New message arrives in conversation

    S->>A: Webhook: New message
    A->>A: Validate webhook signature
    A->>Q: Add job (conversation_id, debounce: 2s)
    Q-->>A: Job ID
    A-->>S: 202 Accepted

    Note over Q: Debounce: Wait 2s for more messages

    Q->>W: Process job
    W->>C: Get conversation context

    alt Cache Hit
        C-->>W: Return cached messages (last 50)
    else Cache Miss
        W->>S: SELECT * FROM messages WHERE conversation_id = ?
        S-->>W: Message history
        W->>C: SET conv:{id} with TTL 1h
    end

    W->>W: Build classification prompt
    W->>V: Classify conversation
    V-->>W: {classification, confidence, reasoning}

    par Parallel Updates
        W->>C: Update cache with classification
        W->>S: UPDATE conversations SET ai_classification_type = ?
        W->>S: INSERT INTO ai_classification_history
    end

    W->>Q: ACK job complete

    Note over S: Conversation now has AI classification
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Data Sources"
        IG[Instagram]
        WA[WhatsApp]
        TT[TikTok]
        FB[Facebook]
        EM[Email]
    end

    subgraph "Existing Platform"
        UI[Unified Inbox]
        DB[(Supabase)]
    end

    subgraph "AI Classifier"
        CL[Classifier]
        CA[Cache]
    end

    subgraph "Output"
        LEAD[Lead]
        SPAM[Spam]
        INFO[Info Needed]
    end

    IG --> UI
    WA --> UI
    TT --> UI
    FB --> UI
    EM --> UI
    UI --> DB
    DB -->|Webhook| CL
    CL <-->|Context| CA
    CL --> LEAD
    CL --> SPAM
    CL --> INFO
    LEAD --> DB
    SPAM --> DB
    INFO --> DB
```

---

## Component Diagram

```mermaid
flowchart TB
    subgraph "API Layer"
        direction TB
        R[Router]
        HE[Health Endpoints]
        WE[Webhook Endpoint]
        CE[Classify Endpoint]
        AE[Admin Endpoints]
        R --> HE
        R --> WE
        R --> CE
        R --> AE
    end

    subgraph "Service Layer"
        direction TB
        CS[Classifier Service]
        VAC[Vertex AI Client]
        SBC[Supabase Client]
        RDC[Redis Client]
        CC[Conversation Cache]
        CS --> VAC
        CS --> SBC
        CS --> CC
        CC --> RDC
        CC --> SBC
    end

    subgraph "Worker Layer"
        direction TB
        QW[Queue Worker]
        JP[Job Processor]
        QW --> JP
        JP --> CS
    end

    WE --> QW
    CE --> CS
```

---

## Deployment Architecture

```mermaid
flowchart TB
    subgraph "Google Cloud Platform"
        subgraph "Cloud Run"
            API[ai-classifier-api<br/>Min: 1 / Max: 10]
            WORKER[ai-classifier-worker<br/>Min: 2 / Max: 20]
        end

        subgraph "Cloud Memorystore"
            REDIS[(Redis<br/>Basic 1GB)]
        end

        subgraph "Vertex AI"
            GEMINI[Gemini 1.5 Flash]
        end

        subgraph "Cloud Build"
            CB[Build Pipeline]
        end

        subgraph "Secret Manager"
            SM[Secrets]
        end

        subgraph "Cloud Monitoring"
            MON[Metrics & Alerts]
            LOG[Cloud Logging]
        end
    end

    subgraph "External"
        SUPA[(Supabase)]
        GH[GitHub]
    end

    GH -->|Push| CB
    CB -->|Deploy| API
    CB -->|Deploy| WORKER
    API --> REDIS
    WORKER --> REDIS
    WORKER --> GEMINI
    API --> SM
    WORKER --> SM
    API --> SUPA
    WORKER --> SUPA
    API --> MON
    WORKER --> MON
    API --> LOG
    WORKER --> LOG
```

---

## Cache Strategy Diagram

```mermaid
flowchart TB
    subgraph "Request Flow"
        REQ[New Message Request]
    end

    subgraph "Cache Layer"
        CHECK{Cache<br/>Exists?}
        HIT[Cache Hit]
        MISS[Cache Miss]
    end

    subgraph "Database"
        FETCH[Fetch from Supabase]
    end

    subgraph "Cache Operations"
        SET[SET with TTL 1h]
        UPDATE[Update Cache]
    end

    subgraph "Classification"
        CLASSIFY[Vertex AI]
        RESULT[Classification Result]
    end

    REQ --> CHECK
    CHECK -->|Yes| HIT
    CHECK -->|No| MISS
    MISS --> FETCH
    FETCH --> SET
    SET --> CLASSIFY
    HIT --> CLASSIFY
    CLASSIFY --> RESULT
    RESULT --> UPDATE
```

---

## Error Handling Flow

```mermaid
flowchart TB
    JOB[Classification Job]
    PROCESS[Process Job]

    PROCESS -->|Success| ACK[Acknowledge]
    PROCESS -->|Transient Error| RETRY{Retry<br/>Count < 3?}
    PROCESS -->|Permanent Error| DLQ[Dead Letter Queue]

    RETRY -->|Yes| BACKOFF[Exponential Backoff]
    RETRY -->|No| DLQ
    BACKOFF --> PROCESS

    DLQ --> ALERT[Alert Team]
    DLQ --> LOG[Log Error Details]

    JOB --> PROCESS
```

---

## Future: Auto-Reply Architecture

```mermaid
flowchart LR
    subgraph "Classification"
        CL[Classifier]
        RES[Classification Result]
    end

    subgraph "Reply Generation"
        RG[Reply Generator]
        VA2[Vertex AI]
    end

    subgraph "Storage"
        DB[(Supabase)]
        SR[Suggested Replies]
    end

    subgraph "UI"
        INBOX[Doctor's Inbox]
    end

    CL --> RES
    RES -->|If Lead| RG
    RG --> VA2
    VA2 --> SR
    SR --> DB
    DB --> INBOX
    INBOX -->|Doctor selects| SEND[Send Reply]
```

---

## Future: Data Extraction Pipeline

```mermaid
flowchart TB
    subgraph "Input"
        CONV[Conversation]
    end

    subgraph "Extraction"
        EXT[Entity Extractor]
        VA3[Vertex AI]
    end

    subgraph "Entities"
        NAME[Name]
        PHONE[Phone]
        EMAIL[Email]
        INTENT[Intent]
        TIME[Preferred Time]
    end

    subgraph "Output"
        DB[(Supabase)]
        CRM[CRM Sync]
    end

    CONV --> EXT
    EXT --> VA3
    VA3 --> NAME
    VA3 --> PHONE
    VA3 --> EMAIL
    VA3 --> INTENT
    VA3 --> TIME
    NAME --> DB
    PHONE --> DB
    EMAIL --> DB
    INTENT --> DB
    TIME --> DB
    DB --> CRM
```
