from nltk.corpus import movie_reviews;
from nltk.classify import NaiveBayesClassifier;
from nltk.classify.util import accuracy as nltk_accuracy
import nltk
nltk.download('movie_reviews')
fileids_pos = movie_reviews.fileids('pos')
fileids_neg = movie_reviews.fileids('neg')

def extract_features(words):
    return dict([(word, True) for word in words])
features_pos = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in fileids_pos]
features_neg = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in fileids_neg]
num_pos = int(len(features_pos) * 0.8)
num_neg = int(len(features_pos) * 0.8)

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]
classifier = NaiveBayesClassifier.train(features_train)

acc = nltk_accuracy(classifier, features_test)
print(f"Точность на тестовой выборке: {acc:.4f} ({acc*100:.2f}%)")

print("\nТоп-20 информативных признаков:")
classifier.show_most_informative_features(20)


test_reviews = [
    "This movie was absolutely fantastic, the acting was brilliant and the plot kept me on the edge of my seat.",
    "I hated every minute of it, the story was boring and the actors were terrible.",
    "A masterpiece of modern cinema, beautifully shot and deeply moving.",
    "Waste of time, poorly written and badly directed.",
    "An enjoyable film with great performances and a clever script.",
    "The movie was dull, predictable, and completely forgettable.",
    "I was impressed by the stunning visuals and the emotional depth of the story.",
    "It was a total disaster, I regret watching it.",
    "A wonderful experience, highly recommended for all audiences.",
    "The plot made no sense and the ending was disappointing."
]

print("\n" + "=" * 90)
print(f"{'Отзыв':<60}{'Тональность':<14}{'P(Pos)':<10}{'P(Neg)':<10}")
print("=" * 90)

for review in test_reviews:
    words = review.lower().split()
    features = extract_features(words)

    prob_dist = classifier.prob_classify(features)
    p_pos = prob_dist.prob('Positive')
    p_neg = prob_dist.prob('Negative')

    predicted = prob_dist.max()

    preview = review[:57]
    print(f"{preview:<60}{predicted:<14}{p_pos:<10.3f}{p_neg:<10.3f}")