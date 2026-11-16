from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from text_chunker import TextChunker

from nltk.corpus import brown
import nltk
nltk.download('brown')

N_WORDS = 1100
M_SENTS = 5

sents = []
total_words = 0
for sent in brown.sents():
    sents.append(sent)
    total_words += len(sent)
    if total_words >= N_WORDS:
        break

print(f"Прочитано предложений: {len(sents)}")
print(f"Всего слов: {total_words}")

chunks = []
for i in range(0, len(sents), M_SENTS):
    group = sents[i:i + M_SENTS]
    chunk_text = " ".join(" ".join(sent) for sent in group)
    chunks.append(chunk_text)

print(f"Получено чанков: {len(chunks)} (по {M_SENTS} предложений)")

chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}

print(f"Получено чанков: {len(chunks)} (по {M_SENTS} предложений)")
print(f"Ключи словаря: {list(chunk_dict.keys())}")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(chunk_dict.values())

print(f"\nРазмерность терм-документной матрицы: {X.shape}")
print(f"(строк = чанков/документов: {X.shape[0]}, "
      f"столбцов = уникальных термов: {X.shape[1]})")

print("\nТерм-документная матрица (плотный вид):")
print(X.toarray())

terms = vectorizer.get_feature_names_out()
print(f"\nПервые 30 термов: {list(terms[:30])}")