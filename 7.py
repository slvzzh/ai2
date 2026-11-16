from nltk import NaiveBayesClassifier;
from nltk.classify import accuracy as nltk_accuracy;
from nltk.corpus import names
import nltk
import random
nltk.download('names')
male_list = [(name, 'male') for name in names.words('male.txt')];
female_list = [(name, 'female') for name in names.words('female.txt')]
data = (male_list + female_list)

def cutWord(word, N):
    return {'ending' : word[-N:]}

print(f"Мужских имён: {len(male_list)}")
print(f"Женских имён: {len(female_list)}")
print(f"Всего: {len(data)}")

random.seed(42)
random.shuffle(data)

test_names = [
    'Alexander', 'Maria', 'Vladimir', 'Anna', 'Dmitry',
    'Ekaterina', 'Sergey', 'Olga', 'Michael', 'Natasha',
    'Ivan', 'Svetlana', 'John', 'Elizabeth', 'Pavel'
]

for N in range(1, 6):
    print("\n" + "=" * 60)
    print(f"N = {N} (учитываем последние {N} букв(ы) имени)")
    print("=" * 60)

    feature_sets = [(cutWord(name, N), gender) for name, gender in data]
    split = int(len(feature_sets) * 0.8)
    train_set = feature_sets[:split]
    test_set = feature_sets[split:]

    classifier = NaiveBayesClassifier.train(train_set)

    acc = nltk_accuracy(classifier, test_set)
    print(f"Точность на тестовой выборке: {acc:.4f} ({acc*100:.2f}%)")

    print(f"\n{'Имя':<15}{'Предсказание':<15}")
    print("-" * 30)
    for name in test_names:
        features = cutWord(name, N)
        prediction = classifier.classify(features)
        print(f"{name:<15}{prediction:<15}")

    print("\nТоп-5 информативных признаков:")
    classifier.show_most_informative_features(5)
