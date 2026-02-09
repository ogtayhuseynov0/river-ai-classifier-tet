-- Prompt Engineering Workbench Tables
-- Run in RESULTS Supabase instance (RESULTS_SUPABASE_URL)

-- ============================================================================
-- Prompt Presets - Reusable prompt configurations
-- ============================================================================

CREATE TABLE IF NOT EXISTS prompt_presets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  name TEXT NOT NULL UNIQUE,
  description TEXT,
  tags TEXT[],

  -- Prompt config
  prompt_type TEXT NOT NULL,          -- 'classification' | 'extraction' | 'response'
  prompt_template TEXT NOT NULL,

  -- Variables toggle map: {"clinic_name": true, "formatted_services": false, ...}
  variables_config JSONB NOT NULL DEFAULT '{}',

  -- Model config
  model_name TEXT NOT NULL DEFAULT 'gemini-2.5-flash-lite',
  temperature DECIMAL(3,2) DEFAULT 0.1,
  top_p DECIMAL(3,2) DEFAULT 0.8,
  max_output_tokens INTEGER DEFAULT 256,
  response_mime_type TEXT DEFAULT 'application/json',

  -- Optional brand DNA key (for response type)
  brand_dna_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_prompt_presets_type ON prompt_presets(prompt_type);
CREATE INDEX IF NOT EXISTS idx_prompt_presets_name ON prompt_presets(name);

-- ============================================================================
-- Test Runs - Every prompt execution is logged
-- ============================================================================

CREATE TABLE IF NOT EXISTS test_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),

  -- Preset reference (nullable - can run without preset)
  preset_id UUID REFERENCES prompt_presets(id) ON DELETE SET NULL,
  preset_name TEXT,

  -- What was tested
  prompt_type TEXT NOT NULL,
  org_id TEXT NOT NULL,
  org_name TEXT,
  thread_id TEXT,
  channel TEXT,

  -- Prompt snapshot (the rendered prompt sent to API)
  prompt_snapshot TEXT NOT NULL,

  -- Model config at time of run
  model_name TEXT NOT NULL,
  temperature DECIMAL(3,2),
  top_p DECIMAL(3,2),
  max_output_tokens INTEGER,
  variables_config JSONB,

  -- Raw API response
  raw_response TEXT,
  parsed_result JSONB,

  -- Classification fields (if prompt_type = 'classification')
  classification TEXT,
  confidence DECIMAL(3,2),
  reasoning TEXT,
  key_signals TEXT[],

  -- Response fields (if prompt_type = 'response')
  generated_response TEXT,
  brand_dna_key TEXT,

  -- Performance
  latency_ms INTEGER,
  token_count INTEGER,

  -- Human feedback
  rating INTEGER,       -- 1 = thumbs up, -1 = thumbs down
  note TEXT
);

CREATE INDEX IF NOT EXISTS idx_test_runs_thread ON test_runs(thread_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_org ON test_runs(org_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_preset ON test_runs(preset_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_created ON test_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_test_runs_type ON test_runs(prompt_type);

-- ============================================================================
-- Triggers
-- ============================================================================

CREATE OR REPLACE FUNCTION update_prompt_presets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS prompt_presets_updated_at ON prompt_presets;
CREATE TRIGGER prompt_presets_updated_at
  BEFORE UPDATE ON prompt_presets
  FOR EACH ROW
  EXECUTE FUNCTION update_prompt_presets_updated_at();
