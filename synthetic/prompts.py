"""Gemini prompt templates for synthetic data generation."""

# =============================================================================
# BUSINESS GENERATION PROMPT
# =============================================================================

BUSINESS_GENERATION_PROMPT = """You are generating synthetic business data for testing purposes.

User request: "{user_prompt}"

Generate a realistic business based on this description. Include:
1. A creative, realistic business name
2. The business type/category
3. 5-10 services with realistic prices
4. A location (city, country)

If the user specified services, use those. Otherwise, generate appropriate services for this business type.

IMPORTANT: Output ONLY valid JSON, no other text. Use this exact format:
{{
  "name": "Business Name Here",
  "business_type": "type of business",
  "location": "City, Country",
  "services": [
    {{"name": "Service 1", "price": 100, "currency": "USD", "description": "Brief description"}},
    {{"name": "Service 2", "price": 200, "currency": "USD", "description": "Brief description"}}
  ]
}}"""


# =============================================================================
# CONTACT GENERATION PROMPT
# =============================================================================

CONTACT_GENERATION_PROMPT = """Generate {num_contacts} realistic synthetic contacts/leads for a {business_type}.

Each contact should have a persona that determines their behavior:
{persona_descriptions}

Generate diverse names (multi-cultural), with varying amounts of demographic info.
Some contacts may have incomplete information (null values).

IMPORTANT: Output ONLY valid JSON array, no other text:
[
  {{
    "first_name": "John",
    "last_name": "Smith",
    "persona": "serious_lead",
    "demographics": {{
      "city": "New York",
      "country": "USA",
      "language": "en",
      "occupation": "Software Engineer"
    }}
  }},
  {{
    "first_name": "Maria",
    "last_name": null,
    "persona": "vague",
    "demographics": {{}}
  }}
]"""

PERSONA_DESCRIPTIONS = """
- serious_lead: Ready to book, provides full details, asks about availability
- curious_lead: Interested but asking questions, comparing options
- price_shopper: Focused on costs, asking for quotes, may negotiate
- spam: Sending promotional content, scams, or irrelevant messages
- wrong_number: Misdirected messages, looking for different business
- vague: Very short messages, unclear intent, just "hi" or "hello"
"""


# =============================================================================
# CONVERSATION GENERATION PROMPT
# =============================================================================

CONVERSATION_GENERATION_PROMPT = """Generate a realistic {channel} conversation between a customer and a business.

BUSINESS INFO:
- Name: {business_name}
- Type: {business_type}
- Services: {services_list}

CUSTOMER INFO:
- Name: {contact_name}
- Persona: {persona}
- Persona behavior: {persona_behavior}

TARGET CLASSIFICATION: {ground_truth}
SCENARIO TYPE: {scenario_type}
NUMBER OF MESSAGES: {num_messages} (alternating customer/business)

CHANNEL STYLE ({channel}):
{channel_style}

REQUIREMENTS:
1. Generate exactly {num_messages} messages, alternating between customer and business
2. First message MUST be from customer
3. Make the conversation clearly classifiable as {ground_truth}
4. Follow the channel style guide for realistic formatting
5. Include realistic timestamps (ISO format) with appropriate gaps

For {ground_truth}:
{ground_truth_guidance}

IMPORTANT: Output ONLY valid JSON array, no other text:
[
  {{"role": "customer", "content": "message text", "timestamp": "2024-01-15T10:30:00Z"}},
  {{"role": "business", "content": "response text", "timestamp": "2024-01-15T10:32:00Z"}}
]"""


# =============================================================================
# CHANNEL STYLES
# =============================================================================

CHANNEL_STYLES = {
    "instagram": """
- Short, casual messages (1-2 sentences max)
- Use emojis sparingly (1-2 per message)
- May reference "saw your post", "found you on IG", "your page"
- Informal greetings: "heyy", "hi there", "hey!"
- Common typos/abbreviations: "thnks", "ur", "rn", "lmk"
- Quick responses expected
""",

    "whatsapp": """
- Conversational, natural flow
- May reference voice notes: "can I send a voice note?"
- More personal, like texting a friend
- Greetings: "Hi", "Hello", "Good morning"
- May use emojis moderately
- Can mention "forwarded this from..."
- Quick back-and-forth expected
""",

    "facebook": """
- Similar to Instagram but slightly more formal
- May reference "saw your Facebook page", "read your reviews"
- Can mention recommendations from friends
- Moderate emoji use
- May include longer messages
- Reference to business hours, location
""",

    "email": """
- More formal structure
- Proper greeting: "Dear [Business]", "Hello"
- Complete sentences, proper grammar
- May include subject line context
- Sign-off: "Best regards", "Thank you", "Sincerely"
- Fewer typos, more professional tone
- Can be longer, more detailed
"""
}


# =============================================================================
# GROUND TRUTH GUIDANCE
# =============================================================================

GROUND_TRUTH_GUIDANCE = {
    "lead": """
- Customer shows clear interest in booking/purchasing a service
- Asks about availability, pricing, or scheduling
- Provides or is willing to provide contact information
- May mention specific services they want
- Shows intent to take action (book, visit, buy)
""",

    "not_lead": """
- Customer is NOT interested in the business services
- Examples: spam messages, job seekers, vendors selling products
- Wrong number/misdirected messages
- Complaints about unrelated matters
- No potential for conversion to a customer
""",

    "needs_info": """
- Not enough information to determine intent
- Very short messages like just "hi" or "hello"
- Vague inquiries without clear purpose
- Conversation ended before intent was clear
- Could go either way with more context
"""
}


# =============================================================================
# PERSONA BEHAVIORS
# =============================================================================

PERSONA_BEHAVIORS = {
    "serious_lead": "Ready to book, provides details like name and preferred times, asks specific questions about the service, wants to schedule an appointment",

    "curious_lead": "Interested but not committed, asking questions about how things work, comparing options, wants more information before deciding",

    "price_shopper": "Focused on costs, asks 'how much is...', wants price list, may try to negotiate, compares with competitors",

    "spam": "Sending promotional content, trying to sell something, irrelevant links, scam messages, not interested in the business services",

    "wrong_number": "Looking for a different business, asking about services not offered, confused about what business this is",

    "vague": "Very minimal messages, just 'hi' or 'hello', doesn't respond with details, unclear what they want"
}


def get_conversation_prompt(
    channel: str,
    business_name: str,
    business_type: str,
    services_list: str,
    contact_name: str,
    persona: str,
    ground_truth: str,
    scenario_type: str,
    num_messages: int
) -> str:
    """Build the full conversation generation prompt."""
    return CONVERSATION_GENERATION_PROMPT.format(
        channel=channel,
        business_name=business_name,
        business_type=business_type,
        services_list=services_list,
        contact_name=contact_name,
        persona=persona,
        persona_behavior=PERSONA_BEHAVIORS.get(persona, "general customer inquiry"),
        ground_truth=ground_truth,
        scenario_type=scenario_type,
        num_messages=num_messages,
        channel_style=CHANNEL_STYLES.get(channel, CHANNEL_STYLES["whatsapp"]),
        ground_truth_guidance=GROUND_TRUTH_GUIDANCE.get(ground_truth, "")
    )


def get_business_prompt(user_prompt: str) -> str:
    """Build the business generation prompt."""
    return BUSINESS_GENERATION_PROMPT.format(user_prompt=user_prompt)


def get_contacts_prompt(num_contacts: int, business_type: str, personas: dict[str, float]) -> str:
    """Build the contacts generation prompt."""
    # Build persona descriptions with counts
    persona_lines = []
    for persona, ratio in personas.items():
        count = max(1, int(num_contacts * ratio))
        desc = PERSONA_BEHAVIORS.get(persona, "general inquiry")
        persona_lines.append(f"- {persona} ({count}): {desc}")

    return CONTACT_GENERATION_PROMPT.format(
        num_contacts=num_contacts,
        business_type=business_type,
        persona_descriptions="\n".join(persona_lines)
    )
