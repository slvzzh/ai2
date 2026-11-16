from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
import nltk
import re

nltk.download('stopwords')

with open('lab02_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Длина текста: {len(text)} символов")

tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{2,}\b')
raw_tokens = tokenizer.tokenize(text.lower())
print(f"Токенов после токенизации: {len(raw_tokens)}")

stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer('english')

def preprocess(tokens):
    """Удаляет стоп-слова и стеммит токены."""
    return [stemmer.stem(t) for t in tokens if t not in stop_words]

sentences_raw = re.split(r'[.!?]+', text)
sentences_raw = [s.strip() for s in sentences_raw if s.strip()]

tokens = []
for sent in sentences_raw:
    sent_tokens = tokenizer.tokenize(sent.lower())
    processed = preprocess(sent_tokens)
    if len(processed) >= 3:   # игнорируем слишком короткие «предложения»
        tokens.append(processed)

print(f"Документов (предложений) для LDA: {len(tokens)}")
print(f"Пример первого документа: {tokens[0][:10]}")

dict_tokens = corpora.Dictionary(tokens)
print(f"Размер словаря: {len(dict_tokens)} уникальных токенов")

doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]
print(f"Терм-документная матрица: {len(doc_term_mat)} документов")

num_topics = 2
ldamodel = models.ldamodel.LdaModel(
    doc_term_mat,
    num_topics=num_topics,
    id2word=dict_tokens,
    passes=25,
    random_state=42
)

print("\n" + "=" * 70)
print(f"Темы LDA (num_topics={num_topics}, passes=25)")
print("=" * 70)

for topic_id in range(num_topics):
    print(f"\nТема {topic_id + 1}:")
    for word, weight in ldamodel.show_topic(topic_id, topn=5):
        print(f"  {word:<20} {weight:.4f}")