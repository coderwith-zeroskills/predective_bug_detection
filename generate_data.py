import pandas as pd
import random

random.seed(42)

# --- Vocabulary pools ---

high_risk_titles = [
    "Migrate user authentication to OAuth2",
    "Refactor payment processing module",
    "Integrate third-party API for bank transfers",
    "Overhaul database schema for orders table",
    "Implement async job queue for email delivery",
    "Replace legacy session management system",
    "Add multi-currency support to checkout",
    "Rewrite data pipeline for real-time analytics",
    "Integrate Stripe webhook for subscription billing",
    "Refactor core user permissions model",
    "Implement distributed caching with Redis",
    "Migrate monolith service to microservices",
    "Add two-factor authentication flow",
    "Overhaul search indexing with Elasticsearch",
    "Implement rate limiting on all API endpoints",
    "Rewrite file upload service with S3 integration",
    "Add GDPR data deletion pipeline",
    "Overhaul notification delivery system",
    "Integrate ML model for fraud detection",
    "Refactor tax calculation engine",
]

safe_titles = [
    "Update button color on landing page",
    "Fix typo in welcome email",
    "Add tooltip to help icon",
    "Update footer copyright year",
    "Increase font size on mobile nav",
    "Rename menu item from Settings to Preferences",
    "Add loading spinner to submit button",
    "Fix broken link in FAQ page",
    "Update placeholder text in search bar",
    "Add aria-label to icon buttons",
    "Change default sort order on listings page",
    "Update privacy policy text",
    "Fix alignment issue in dashboard cards",
    "Add hover state to sidebar links",
    "Update error message wording",
    "Fix padding inconsistency in modal",
    "Add character counter to bio field",
    "Update onboarding copy for clarity",
    "Fix z-index issue with dropdown menu",
    "Add confirmation dialog before logout",
]

high_risk_descriptions = [
    "This involves changes to the core authentication flow. Multiple services depend on this module. Needs thorough testing across all environments.",
    "Significant refactor touching payment logic, error handling, and third-party API contracts. Risk of regression in billing edge cases.",
    "Requires schema migration on a production table with 10M+ rows. Downtime window needed. Rollback plan must be prepared.",
    "Async processing introduces race conditions and retry logic. Needs careful testing under load.",
    "Touches sensitive PII data. Must comply with GDPR deletion requirements across 3 microservices.",
    "Integration with external vendor API — subject to rate limits, versioning, and unstable sandbox environment.",
    "Multi-currency conversion involves floating point edge cases and tax rule variations by region.",
    "Cache invalidation strategy must be carefully designed to avoid stale data issues.",
    "Webhook processing requires idempotency handling to prevent duplicate charge events.",
    "Permission model change will affect all role-based access checks across the application.",
]

safe_descriptions = [
    "Simple copy change on the marketing page. No backend changes required.",
    "CSS-only fix. No logic changes. Can be verified visually in staging.",
    "Minor UI update. Only affects the component in isolation.",
    "Static text update. No database or API changes involved.",
    "Accessibility improvement. Adding labels to existing elements.",
    "Cosmetic fix requested by design team. Low effort, low risk.",
    "Small UX improvement. No functional behavior changes.",
    "Wording update approved by legal. No code logic involved.",
    "Padding fix in a single component. Isolated change.",
    "Tooltip added to clarify existing feature. No new functionality.",
]

high_risk_keywords = [
    "authentication", "payment", "migration", "async", "webhook",
    "encryption", "OAuth", "database schema", "third-party", "PII",
    "race condition", "rollback", "compliance", "distributed", "refactor",
    "legacy", "permissions", "real-time", "fraud", "billing"
]

safe_keywords = [
    "typo", "color", "font", "tooltip", "alignment",
    "copy", "placeholder", "label", "hover", "padding",
    "icon", "wording", "modal", "dropdown", "onboarding"
]

ambiguous_phrases = [
    "Update login page UI",
    "Fix styling on payment confirmation screen",
    "Add loading state to auth button",
    "Improve error message on checkout failure",
    "Refactor tooltip on permissions settings",
    "Clean up legacy CSS in user profile",
    "Update copy on OAuth consent screen",
]

noise_risk_words = [
    "database", "cache", "token", "session",
    "timeout", "retry", "concurrency", "deployment"
]
# --- Story generation ---

def make_story(is_high_risk):
    if is_high_risk:
        title = random.choice(high_risk_titles)
        # 30% chance: swap in an ambiguous title to create overlap
        if random.random() < 0.3:
            title = random.choice(ambiguous_phrases)
        base_desc = random.choice(high_risk_descriptions)
        extra_keywords = random.sample(high_risk_keywords, k=random.randint(2, 4))
        description = base_desc + " Key concerns: " + ", ".join(extra_keywords) + "."
        story_points = random.choice([5, 8, 13, 21])
        num_comments = random.randint(3, 20)
        # 20% chance: low story points despite being risky (noise)
        if random.random() < 0.2:
            story_points = random.choice([1, 2, 3])
            num_comments = random.randint(0, 3)
        label = 1
    else:
        title = random.choice(safe_titles)
        # 25% chance: safe story accidentally contains a risky-sounding word
        if random.random() < 0.25:
            title = random.choice(ambiguous_phrases)
        base_desc = random.choice(safe_descriptions)
        extra_keywords = random.sample(safe_keywords, k=random.randint(1, 3))
        # 15% chance: safe story mentions a noise risk word in description
        if random.random() < 0.15:
            noise_word = random.choice(noise_risk_words)
            base_desc += f" Note: involves minor {noise_word} consideration."
        description = base_desc + " Notes: " + ", ".join(extra_keywords) + "."
        story_points = random.choice([1, 2, 3])
        num_comments = random.randint(0, 5)
        # 15% chance: safe story has higher points (estimation noise)
        if random.random() < 0.15:
            story_points = random.choice([5, 8])
        label = 0

    return {
        "title": title,
        "description": description,
        "story_points": story_points,
        "num_comments": num_comments,
        "label": label
    }

# --- Build dataset ---

stories = []

# 250 high risk, 250 safe
for _ in range(250):
    stories.append(make_story(is_high_risk=True))

for _ in range(250):
    stories.append(make_story(is_high_risk=False))

# Shuffle so they're not all grouped together
random.shuffle(stories)

df = pd.DataFrame(stories)

# Combine title + description into one text column for NLP
df["full_text"] = df["title"] + " " + df["description"]

# Save to CSV
df.to_csv("jira_stories.csv", index=False)

print(f"Dataset created: {len(df)} stories")
print(f"High risk stories : {df['label'].sum()}")
print(f"Safe stories       : {(df['label'] == 0).sum()}")
print()
print("Sample rows:")
print(df[["title", "story_points", "num_comments", "label"]].head(8).to_string())