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

MAX_CHUNKS = 20
MAX_CHARS = 50

chunks_to_show = chunks[:MAX_CHUNKS]
if len(chunks) > MAX_CHUNKS:
    print(f"\n[Показаны первые {MAX_CHUNKS} из {len(chunks)} чанков]\n")

for i, chunk in enumerate(chunks_to_show, 1):
    preview = chunk[:MAX_CHARS]
    print(f"Чанк {i:>2}: {preview}")

