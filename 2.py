from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
words = [
    "running", "runs", "runner", "easily", "fairly",
    "happiness", "happily", "relational", "conditional", "rational",
    "generously", "generation", "generate", "generalization",
    "studies", "studying", "studied", "university", "universal",
    "maximum", "presumably", "multiply", "oscillating", "oscillation",
    "effective", "effectively", "effectiveness", "connected", "connecting"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

results = []
for w in words:
    results.append((
        w,
        porter.stem(w),
        lancaster.stem(w),
        snowball.stem(w)
    ))

print(f"{'Слово':<16}{'Porter':<16}{'Lancaster':<16}{'Snowball':<16}")
print("-" * 64)
for w, p, l, s in results:
    print(f"{w:<16}{p:<16}{l:<16}{s:<16}")