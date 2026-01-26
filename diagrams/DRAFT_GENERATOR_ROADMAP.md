# Draft Generator - Context Roadmap

## Overview

AI-powered message drafting requires rich context to generate relevant, personalized responses. This document outlines what context is needed and the implementation phases.

---

## Context Sources

### Currently Implemented
| Source | Description |
|--------|-------------|
| Chat History | Last N messages in conversation |
| Business Name/Type | Organization identity |
| Services List | Available services |
| Brand DNA | Tone, phrases, style rules |
| Channel Type | Instagram, WhatsApp, Email, etc. |

### Planned
| Source | What It Provides | Example Use |
|--------|------------------|-------------|
| Calendar Access | Available slots, busy times | "We have openings Tuesday at 2pm and 4pm" |
| Contact History | Past appointments, notes, purchases | "Welcome back! Last time you had teeth whitening" |
| Booking Links/Forms | Channel-specific URLs | "Book here: [link]" |
| Business Hours/Rules | Open times, holidays, policies | "We're closed Mondays" |
| Promotions/Offers | Current deals | "We have 20% off this week" |
| Staff Info | Who does what | "Dr. Smith handles implants" |
| FAQ/Knowledge Base | Common Q&A | Pre-written answers to frequent questions |
| Booking Rules | Min notice, cancellation policy | "We require 24hr notice" |
| Contact Preferences | Language, channel preference | Respond in preferred language |
| Lead Score/Stage | Hot lead vs browsing | Adjust response urgency |

---

## Scalability: Tiered Context Loading

Can't send everything to LLM every time due to:
- Token limits
- Latency (more context = slower)
- Cost (more tokens = $$$)

### Tier 1: Always Include (~500 tokens)
- Business name/type
- Brand DNA
- Last 5 messages
- Top 5 relevant services

### Tier 2: Include if Available (~300 tokens)
- Contact summary (returning customer? past services?)
- Current promotions
- Business hours

### Tier 3: Intent-Based (~200 tokens)
| If Intent | Include |
|-----------|---------|
| Booking | Calendar availability, booking link |
| Price inquiry | Service prices, promotions |
| Complaint | Escalation rules, manager contact |
| FAQ match | Pre-written answer |

---

## Implementation Phases

### Phase 1: Foundation
**Status:** In Progress

- [x] Chat history
- [x] Services list
- [x] Brand DNA profiles
- [ ] Business config (hours, rules, links)

**Deliverable:** JSON config per organization with basic business rules.

---

### Phase 2: Contact Intelligence
**Status:** Planned

- [ ] Contact history summary
  - Last appointment date
  - Total visits
  - Services used before
  - Notes/tags
- [ ] Lead stage awareness (new lead, returning, VIP)
- [ ] Preferred language detection
- [ ] Contact sentiment (happy, frustrated, neutral)

**Deliverable:** Contact context injected into prompts for personalized responses.

---

### Phase 3: Smart Scheduling
**Status:** Planned

- [ ] Calendar integration (read-only)
  - Available slots
  - Staff availability
  - Service duration awareness
- [ ] Booking link injection (per channel)
- [ ] Time zone handling
- [ ] Appointment type matching

**Deliverable:** AI can suggest specific available times and provide booking links.

---

### Phase 4: Knowledge & Automation
**Status:** Planned

- [ ] FAQ/Knowledge base
  - Vector search for relevant answers
  - Pre-approved response templates
- [ ] Intent detection
  - Booking intent
  - Price inquiry
  - Complaint/issue
  - General question
- [ ] Auto-select relevant context based on detected intent
- [ ] Response confidence scoring

**Deliverable:** Smarter context selection, faster responses, lower token usage.

---

### Phase 5: Advanced Features
**Status:** Future

- [ ] Staff routing (match inquiry to right person)
- [ ] Promotion engine (auto-suggest relevant offers)
- [ ] Multi-location support
- [ ] Compliance guardrails (HIPAA, GDPR)
- [ ] A/B testing different response styles
- [ ] Response analytics (which drafts get accepted?)

**Deliverable:** Enterprise-ready draft generation system.

---

## Architecture

```
Customer Message
       │
       ▼
┌──────────────────┐
│ Intent Detector  │  ← "What does customer want?"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Context Selector │  ← "What context is needed?"
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Cache │ │  DB   │   ← Fetch only needed context
└───┬───┘ └───┬───┘
    └────┬────┘
         │
         ▼
┌──────────────────┐
│ Context Builder  │  ← Assemble prompt (stay under token limit)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   LLM (Gemini)   │
└────────┬─────────┘
         │
         ▼
   Draft Response
```

---

## Quick Reference: Organization Config Schema

```json
{
  "org_id": "uuid",
  "business_hours": {
    "monday": "9:00-18:00",
    "tuesday": "9:00-18:00",
    "wednesday": "9:00-18:00",
    "thursday": "9:00-18:00",
    "friday": "9:00-18:00",
    "saturday": "10:00-14:00",
    "sunday": "closed"
  },
  "booking_links": {
    "default": "https://clinic.com/book",
    "instagram": "https://clinic.com/book?ref=ig",
    "whatsapp": "https://wa.me/1234567890"
  },
  "policies": {
    "cancellation": "24 hours notice required",
    "deposits": "50% deposit for treatments over $500"
  },
  "current_promotions": [
    {
      "name": "January Whitening Special",
      "description": "20% off teeth whitening",
      "valid_until": "2026-01-31"
    }
  ],
  "quick_responses": {
    "after_hours": "Thanks for your message! We're currently closed but will respond first thing tomorrow.",
    "emergency": "For dental emergencies, please call our emergency line: 555-1234"
  },
  "brand_dna": {
    "tone": "friendly, professional",
    "personality": "helpful healthcare provider",
    "use_emojis": false,
    "preferred_phrases": ["Happy to help", "Looking forward to seeing you"],
    "avoid_phrases": ["Unfortunately", "We can't"]
  }
}
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Draft acceptance rate | >70% |
| Time to first response | <30 seconds |
| Token usage per draft | <1000 tokens |
| Context relevance score | >85% |
