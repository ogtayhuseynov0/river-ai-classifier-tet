# Synthetic Data Generation

Prompt-driven synthetic data generator for ANY business type using Gemini AI.

---

## Completed

- [x] `synthetic/__init__.py` - Module exports
- [x] `synthetic/config.py` - GenerationConfig dataclass
- [x] `synthetic/models.py` - SyntheticBusiness, SyntheticContact, SyntheticConversation, SyntheticDataset
- [x] `synthetic/prompts.py` - Gemini prompt templates for business/contact/conversation generation
- [x] `synthetic/generator.py` - SyntheticDataGenerator class with CLI
- [x] `synthetic/exporter.py` - JSON/CSV export utilities
- [x] `app.py` integration - Sidebar UI for generating synthetic data

---

## Usage

### Python API
```python
from synthetic import SyntheticDataGenerator, GenerationConfig

generator = SyntheticDataGenerator()
config = GenerationConfig(num_conversations=50)

# Generate for ANY business type
dataset = generator.generate(
    prompt="Barbershop in Brooklyn with haircuts and beard trims",
    config=config
)

# Or law firm, yacht charter, dental clinic, etc.
dataset = generator.generate(
    prompt="Law firm specializing in immigration and family law",
    config=config
)

# Export
dataset.to_json("output.json")
dataset.to_csv("output.csv")
```

### CLI
```bash
python -m synthetic.generator "Barbershop in Brooklyn" --num-conversations 20
```

### Streamlit UI
1. Run `streamlit run app.py`
2. Open sidebar → "🧪 Synthetic Data" → "Generate Test Data"
3. Enter business description and generate

---

## File Structure

```
synthetic/
├── __init__.py          # Module exports
├── config.py            # GenerationConfig
├── models.py            # Data models
├── prompts.py           # Gemini prompts
├── generator.py         # Main generator + CLI
├── exporter.py          # Export utilities
└── output/              # Generated datasets (gitignored)
```

---

## Features

- **Any business type**: lawyers, barbers, yacht services, dentists, etc.
- **AI-generated**: Gemini creates realistic business names, services, contacts, conversations
- **Ground truth labels**: lead, not_lead, needs_info with persona-based scenarios
- **Channel styles**: Instagram, WhatsApp, Facebook, Email with appropriate formatting
- **Reproducible**: Seed option for deterministic generation
- **Export**: JSON (full) and CSV (flat) formats

---

## Future Enhancements

- [ ] Batch classifier accuracy testing
- [ ] Multiple languages support
- [ ] Pre-generated sample datasets
- [ ] Supabase import format
