from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
words = [
    "running", "runs", "runner", "easily", "fairly",
    "happiness", "happily", "relational", "conditional", "rational",
    "generously", "generation", "generate", "generalization",
    "studies", "studying", "studied", "university", "universal",
    "maximum", "presumably", "multiply", "oscillating", "oscillation",
    "effective", "effectively", "effectiveness", "connected", "connecting"
]
results = []
lemmatizer = WordNetLemmatizer()
for w in words:
    results.append((
        w,
        lemmatizer.lemmatize(w, pos='n'),
        lemmatizer.lemmatize(w, pos='v')
    )
    )
print(f"{'Слово':<16}{'Существительное':<16}{'Глагол':<16}")
print("-" * 64)
for w, n, v in results:
    print(f"{w:<16}{n:<16}{v:<16}")