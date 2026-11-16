from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

categories = [
    'alt.atheism',
    'comp.graphics',
    'sci.med',
    'soc.religion.christian',
    'talk.politics.guns'
]

training_data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(training_data.data)

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

classifier = MultinomialNB().fit(train_tfidf, training_data.target)

input_sentences = [
    "The graphics card supports real-time ray tracing and high frame rates.",
    "The patient was diagnosed with a severe infection and prescribed antibiotics.",
    "Faith and prayer are central to a Christian's daily life.",
    "The debate over gun control continues after the recent shooting.",
    "There is no scientific evidence that supports the existence of a deity."
]

input_counts = count_vect.transform(input_sentences)
input_tfidf = tfidf_transformer.transform(input_counts)

predictions = classifier.predict(input_tfidf)

print(f"{'Предложение':<70} {'Предсказанная категория'}")
print("-" * 110)
for sentence, predicted_index in zip(input_sentences, predictions):
    predicted_category = training_data.target_names[predicted_index]
    print(f"{sentence[:70]:<70} {predicted_category}")