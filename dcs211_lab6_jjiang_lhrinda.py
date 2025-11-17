# dcs211_lab6_jjiang_lhrinda

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Tuple

# Download required NLTK resources, to block VADER, add a # in the second line.
nltk.download("punkt_tab")
nltk.download("vader_lexicon")


def extract_features(text: str) -> Dict[str, bool]:
    '''
    Extract bag-of-words features from a review sentence.

    Parameters:
        text: str

    Returns:
        dict: A feature dictionary mapping "contains(word)" to True
    '''
    words = word_tokenize(text.lower())
    return {f"contains({w})": True for w in words}


def build_datasets():
    '''
    Build training and testing datasets.
    Expanded to include more examples for better accuracy.

    Returns:
        train_set (list), test_set (list)
    '''

    # Positive review examples
    positive_sentences = [
        "The pizza was amazing and the service was very friendly.",
        "I loved the noodles, everything tasted fresh and delicious.",
        "Great restaurant, the staff were kind and the food was perfect.",
        "The dessert was fantastic and the atmosphere was cozy.",
        "Our meal was wonderful and we will definitely come back.",
        "Excellent food and outstanding service, highly recommend!",
        "The steak was cooked to perfection and tasted incredible.",
        "Best dining experience ever, the chef is truly talented.",
        "Absolutely delicious meal with beautiful presentation.",
        "Five stars! Everything was exceptional from start to finish.",
    ]

    # Negative review examples
    negative_sentences = [
        "The soup was cold and the waiter was rude.",
        "Terrible experience, the food was bland and overpriced.",
        "I hated the burger, it was dry and tasteless.",
        "Awful service, we waited an hour and the meal was still wrong.",
        "The restaurant was dirty and the staff ignored us.",
        "Worst meal ever, everything was greasy and disgusting.",
        "The fish smelled bad and tasted even worse.",
        "Horrible place, never coming back, waste of money.",
        "Undercooked chicken and unfriendly waiters, very disappointing.",
        "Overpriced garbage, the food was cold and the portions tiny.",
    ]

    # Neutral review examples - expanded to 10 examples
    neutral_sentences = [
        "The restaurant is small and close to the station.",
        "We had lunch at 2 pm, it was not very busy.",
        "The menu has many options and the prices are average.",
        "The place was crowded but the music was okay.",
        "We sat by the window, the view was nice but the food was just fine.",
        "The restaurant is located downtown and has parking available.",
        "It's a typical chain restaurant with standard menu items.",
        "The food was acceptable, nothing special but not bad either.",
        "Average dining experience, met our basic expectations.",
        "Decent place for a quick meal, nothing memorable.",
    ]

    # Convert sentences into (features, label) format
    labeled_data = []

    for sent in positive_sentences:
        labeled_data.append((extract_features(sent), "pos"))
    for sent in negative_sentences:
        labeled_data.append((extract_features(sent), "neg"))
    for sent in neutral_sentences:
        labeled_data.append((extract_features(sent), "neu"))

    # Split: first 8 of each category for training, last 2 for testing
    train_set = (
        labeled_data[0:8]       # pos
        + labeled_data[10:18]   # neg
        + labeled_data[20:28]   # neu
    )
    test_set = (
        labeled_data[8:10]      # pos
        + labeled_data[18:20]   # neg
        + labeled_data[28:30]   # neu
    )

    return train_set, test_set


def train_classifier():
    '''
    Train a Naive Bayes classifier using the constructed dataset.

    Returns:
        classifier (NaiveBayesClassifier)
    '''
    train_set, test_set = build_datasets()
    classifier = NaiveBayesClassifier.train(train_set)

    print("Accuracy on test set:", f"{accuracy(classifier, test_set):.2%}")
    print()
    print("Most informative features:")
    classifier.show_most_informative_features(10)

    return classifier


def classify_review(classifier, text: str) -> str:
    '''
    Classify a single review into pos / neg / neu.

    Parameters:
        classifier: A trained NaiveBayesClassifier.
        text: str, input review sentence.

    Returns:
        str: The predicted label ("pos" / "neg" / "neu").
    '''
    feats = extract_features(text)
    return classifier.classify(feats)


def classify_with_prob(classifier, text: str):
    '''
    Classify a review and show probability distribution.

    Parameters:
        classifier: A trained NaiveBayesClassifier.
        text: str, input review sentence.

    Returns:
        tuple: (predicted_label, probability_dict)
    '''
    feats = extract_features(text)
    prob_dist = classifier.prob_classify(feats)

    # Get probabilities for each label
    probs = {
        'pos': prob_dist.prob('pos'),
        'neg': prob_dist.prob('neg'),
        'neu': prob_dist.prob('neu')
    }

    return prob_dist.max(), probs

# You will figure out our package is not enough for identify the sentiment correctly, here is a free Vader package
# Block it first to see our package, remeber to also block the one in the main fucntion

sia = SentimentIntensityAnalyzer()


def classify_with_vader(text: str) -> Tuple[str, Dict[str, float]]:
    '''
    Classify a review using VADER sentiment lexicon.

    VADER returns four scores:


    We convert the compound score into pos/neg/neu label
    using simple thresholds.

    Parameters:
        text: str, input review sentence.

    Returns:
        tuple: (predicted_label, scores_dict)
            predicted_label: "pos" / "neg" / "neu"
            scores_dict: the full scores from VADER
    '''
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.3:
        label = 'pos'
    elif compound <= -0.3:
        label = 'neg'
    else:
        label = 'neu'

    return label, scores

def main():

    print("Training sentiment classifier...")
    classifier = train_classifier()

    print("Training complete! Enter a 1-2 sentence restaurant review.")
    print("Press Enter on an empty line or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("Enter a review: ").strip()
            if not user_input:
                print("Empty input detected. Exiting program.")
                break

            # Naive Bayes classification (original logic)
            label, probs = classify_with_prob(classifier, user_input)
            print(f"\n[Naive Bayes]")
            print(f"Predicted sentiment: {label}")
            print(f"Confidence: pos={probs['pos']:.1%}, "
                  f"neg={probs['neg']:.1%}, neu={probs['neu']:.1%}")

            # VADER classification 
            vader_label, vader_scores = classify_with_vader(user_input)
            print(f"\n[VADER]")
            print(f"Predicted sentiment: {vader_label}")
            print(
                f"Scores: pos={vader_scores['pos']:.2f}, "
                f"neu={vader_scores['neu']:.2f}, "
                f"neg={vader_scores['neg']:.2f}, "
                f"compound={vader_scores['compound']:.2f}"
            )

            print() 

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting program.")
            break


if __name__ == "__main__":
    main()
