-- Classification Results Table
-- Run this in your RESULTS Supabase instance

CREATE TABLE IF NOT EXISTS classification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Source data references
    org_id TEXT NOT NULL,
    org_name TEXT,
    chat_id TEXT NOT NULL,
    chat_name TEXT,
    contact_id TEXT,
    contact_name TEXT,
    channel TEXT,

    -- Ground truth from source DB
    ground_truth TEXT,  -- Lead, Customer, or null

    -- AI Classification results
    ai_classification TEXT,  -- LEAD, NOT_LEAD, NEEDS_INFO
    confidence DECIMAL(3,2),
    match_status TEXT,  -- Match, Mismatch, etc.
    reasoning TEXT,
    key_signals TEXT[],

    -- Extracted data
    ext_first_name TEXT,
    ext_last_name TEXT,
    ext_date_of_birth TEXT,
    ext_gender TEXT,
    ext_city TEXT,
    ext_country TEXT,
    ext_language TEXT,
    ext_occupation TEXT,
    ext_matched_services JSONB,  -- Array of {service, confidence}
    ext_metadata JSONB,

    -- Messages snapshot
    message_count INTEGER,
    last_5_messages TEXT,

    -- User feedback
    note TEXT,

    -- Indexes
    UNIQUE(chat_id)  -- One result per chat, update on re-classify
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_results_org_id ON classification_results(org_id);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON classification_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_ai_classification ON classification_results(ai_classification);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_classification_results_updated_at
    BEFORE UPDATE ON classification_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
