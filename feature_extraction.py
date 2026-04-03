# we use TF-IDF : https://www.youtube.com/watch?v=D2V1okCEsiE



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
import pickle
import scipy.sparse as sp

# Download stopwords (only runs once)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv("jira_stories.csv")
print(f"Loaded {len(df)} stories")

# ── 2. Text cleaning ───────────────────────────────────────────────────────────

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Lowercase everything
    text = text.lower()
    # Keep only letters and spaces (remove punctuation, numbers)
    text = "".join(ch if ch.isalpha() or ch == " " else " " for ch in text)
    # Remove stopwords (the, is, and, etc.)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["cleaned_text"] = df["full_text"].apply(clean_text)

print("\nCleaning example:")
print("  BEFORE:", df["full_text"].iloc[4][:80])
print("  AFTER :", df["cleaned_text"].iloc[4][:80])

# ── 3. TF-IDF on cleaned text ──────────────────────────────────────────────────
# max_features=500 → keep only the 500 most important words
# ngram_range=(1,2) → capture single words AND two-word phrases
#                     e.g. "database schema" is more meaningful than just "database"

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2          # ignore words that appear in fewer than 2 stories
)

tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])
print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"  → {tfidf_matrix.shape[0]} stories × {tfidf_matrix.shape[1]} word features")

# Show top words learned
feature_names = vectorizer.get_feature_names_out()
print(f"\nSample words TF-IDF learned:")
print(" ", list(feature_names[:20]))

# ── 4. Numeric features ────────────────────────────────────────────────────────
# These are the non-text signals: story points and comment count
# We scale them so they don't overpower the TF-IDF scores

numeric_features = df[["story_points", "num_comments"]].values

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

print(f"\nNumeric features shape: {numeric_scaled.shape}")

# ── 5. Keyword flag features ───────────────────────────────────────────────────
# Hand-crafted signals: does this story mention high-risk words?
# These are called "domain knowledge features" — very valued in DS interviews

risk_keywords = [
    "auth", "payment", "migrat", "async", "webhook",
    "encrypt", "oauth", "schema", "legacy", "permiss",
    "refactor", "fraud", "billing", "compliance", "race"
]

def keyword_flags(text):
    text_lower = text.lower()
    return [1 if kw in text_lower else 0 for kw in risk_keywords]

keyword_matrix = np.array([keyword_flags(t) for t in df["full_text"]])
print(f"Keyword flag features shape: {keyword_matrix.shape}")
print(f"  → 1 column per risk keyword ({len(risk_keywords)} keywords)")

# ── 6. Combine everything into one feature matrix ──────────────────────────────
# Final matrix = TF-IDF (500 cols) + numeric (2 cols) + keywords (15 cols)
# Total = 517 features per story

numeric_sparse   = sp.csr_matrix(numeric_scaled)
keyword_sparse   = sp.csr_matrix(keyword_matrix)

X = sp.hstack([tfidf_matrix, numeric_sparse, keyword_sparse])
y = df["label"].values

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"  → {X.shape[1]} total features per story")
print(f"  → {X.shape[0]} stories")
print(f"\nLabel distribution:")
print(f"  High risk (1): {y.sum()}")
print(f"  Safe      (0): {(y == 0).sum()}")

# ── 7. Save everything for Phase 4 ────────────────────────────────────────────

sp.save_npz("features.npz", X)
np.save("labels.npy", y)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved:")
print("  features.npz   ← feature matrix for training")
print("  labels.npy     ← labels (0 or 1)")
print("  vectorizer.pkl ← TF-IDF model (reuse on new stories)")
print("  scaler.pkl     ← numeric scaler (reuse on new stories)")
print("\nPhase 3 complete.")