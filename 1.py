from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
import nltk
nltk.download('punkt')
text = """Nvidia was founded on April 5, 1993, by Jensen Huang, 
a Taiwanese-American electrical engineer who was previously the 
director of CoreWare at LSI Logic and a microprocessor designer at 
AMD, Chris Malachowsky, an engineer who worked at Sun Microsystems, 
and Curtis Priem, who was previously a senior staff engineer and graphics 
chip designer at IBM and Sun Microsystems. In late 1992, the three men agreed 
to start the company by sketching out their initial ideas in a meeting at a 
Denny's roadside diner on Berryessa Road in East San Jose."""

sentences = sent_tokenize(text, language='english')
print("Количество предложений:", len(sentences))
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")

words = word_tokenize(text, language='english')
print("Количество токенов:", len(words))
print(words)

wp_tokens = WordPunctTokenizer().tokenize(text)
print("Количество токенов:", len(wp_tokens))
print(wp_tokens)