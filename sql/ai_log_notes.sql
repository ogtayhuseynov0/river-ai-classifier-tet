-- AI Log Notes Table
-- Run in RESULTS Supabase instance (RESULTS_SUPABASE_URL)
-- Stores notes for classification logs and draft messages

CREATE TABLE IF NOT EXISTS ai_log_notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  -- Reference to CRM data
  org_id TEXT NOT NULL,
  log_type TEXT NOT NULL,  -- 'classification' or 'draft'
  log_id UUID NOT NULL,    -- ID from classification_logs or draft_messages
  thread_id UUID,          -- For easier querying
  message_id UUID,         -- For easier querying

  -- Note content
  note TEXT,

  UNIQUE(org_id, log_type, log_id)  -- One note per log per org
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ai_log_notes_org ON ai_log_notes(org_id);
CREATE INDEX IF NOT EXISTS idx_ai_log_notes_thread ON ai_log_notes(thread_id);
CREATE INDEX IF NOT EXISTS idx_ai_log_notes_log_type ON ai_log_notes(log_type);

-- Trigger to update updated_at on changes
CREATE OR REPLACE FUNCTION update_ai_log_notes_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ai_log_notes_updated_at ON ai_log_notes;
CREATE TRIGGER ai_log_notes_updated_at
  BEFORE UPDATE ON ai_log_notes
  FOR EACH ROW
  EXECUTE FUNCTION update_ai_log_notes_updated_at();
