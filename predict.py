import pickle
import numpy as np
import scipy.sparse as sp
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── Load all saved artefacts ───────────────────────────────────────────────────

with open("model.pkl",      "rb") as f: model      = pickle.load(f)
with open("vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
with open("scaler.pkl",     "rb") as f: scaler     = pickle.load(f)
with open("threshold.pkl",  "rb") as f: threshold  = pickle.load(f)

stop_words = set(stopwords.words("english"))

RISK_KEYWORDS = [
    "auth", "payment", "migrat", "async", "webhook",
    "encrypt", "oauth", "schema", "legacy", "permiss",
    "refactor", "fraud", "billing", "compliance", "race"
]

# ── Feature extraction (mirrors feature_extraction.py exactly) ─────────────────

def clean_text(text):
    text = text.lower()
    text = "".join(ch if ch.isalpha() or ch == " " else " " for ch in text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def extract_features(title, description, story_points, num_comments):
    full_text   = title + " " + description
    cleaned     = clean_text(full_text)

    # TF-IDF
    tfidf = vectorizer.transform([cleaned])

    # Numeric
    numeric = scaler.transform([[story_points, num_comments]])
    numeric_sparse = sp.csr_matrix(numeric)

    # Keyword flags
    text_lower = full_text.lower()
    flags = np.array([[1 if kw in text_lower else 0 for kw in RISK_KEYWORDS]])
    flags_sparse = sp.csr_matrix(flags)

    return sp.hstack([tfidf, numeric_sparse, flags_sparse])

# ── Predict function ───────────────────────────────────────────────────────────

def predict_risk(title, description, story_points, num_comments):
    X = extract_features(title, description, story_points, num_comments)
    prob = model.predict_proba(X)[0][1]
    is_risky = prob >= threshold

    print("\n" + "=" * 52)
    print("  JIRA STORY RISK ASSESSMENT")
    print("=" * 52)
    print(f"  Title        : {title}")
    print(f"  Story points : {story_points}")
    print(f"  Comments     : {num_comments}")
    print("-" * 52)
    print(f"  Risk score   : {prob:.1%}")
    print(f"  Threshold    : {threshold:.0%}")

    # Visual risk bar
    filled = int(prob * 20)
    bar = "#" * filled + "-" * (20 - filled)
    print(f"  [{bar}]")
    print("-" * 52)

    if prob >= 0.75:
        verdict = "HIGH RISK   — flag before sprint commitment"
        icon = "  [!!!]"
    elif prob >= threshold:
        verdict = "MODERATE RISK — review carefully"
        icon = "  [!]  "
    else:
        verdict = "LOW RISK    — safe to commit"
        icon = "  [OK] "

    print(f"{icon} {verdict}")

    # Which risk keywords were found
    found = [kw for kw in RISK_KEYWORDS if kw in (title + " " + description).lower()]
    if found:
        print(f"\n  Risk signals : {', '.join(found)}")
    else:
        print(f"\n  Risk signals : none detected")

    print("=" * 52)
    return prob

# ── Test stories ───────────────────────────────────────────────────────────────

print("\nRunning predictions on test stories...\n")

# Should be HIGH RISK
predict_risk(
    title="Migrate user authentication to OAuth2",
    description="Core auth system change. All services depend on this. Needs full regression testing.",
    story_points=13,
    num_comments=12
)

# Should be HIGH RISK
predict_risk(
    title="Refactor payment processing module",
    description="Touching billing logic and Stripe webhook integration. Edge cases in multi-currency.",
    story_points=21,
    num_comments=18
)

# Should be LOW RISK
predict_risk(
    title="Update footer copyright year",
    description="Static text change on marketing page. No backend changes required.",
    story_points=1,
    num_comments=0
)

# Should be LOW RISK
predict_risk(
    title="Fix typo in welcome email",
    description="Copy change only. Approved by content team.",
    story_points=1,
    num_comments=1
)

# Tricky: sounds safe but has risk signals
predict_risk(
    title="Update login page button label",
    description="Design team requested label change from Sign In to Log In. CSS only.",
    story_points=2,
    num_comments=1
)

# Tricky: sounds risky but is actually safe
predict_risk(
    title="Add tooltip to OAuth settings page",
    description="Add a small help tooltip next to the existing field. No logic changes.",
    story_points=2,
    num_comments=2
)