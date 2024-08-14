from textblob import TextBlob
from spacy.tokens import Doc
from collections import Counter
import re
import numpy as np


def count_pos_tags(doc: Doc) -> dict[str, int]:
    """Count number of each part-of-speech tag in `doc`"""
    tags = [
        "PUNCT",
        "AUX",
        "PART",
        "VERB",
        "ADP",
        "NUM",
        "NOUN",
        "CCONJ",
        "PRON",
        "ADV",
        "ADJ",
        "PROPN",
        "DET",
        "INTJ",
        "SCONJ",
        "SPACE",
        "SYM",
        "X",
    ]
    pos_tags = {}
    for tag in tags:
        pos_tags[tag] = 0
    for token in doc:
        pos = token.pos_
        if pos in pos_tags:
            pos_tags[pos] += 1
    return pos_tags


def count_hapaxes(doc: Doc) -> int:
    """Count the number of hapaxes (words that only occur once) in `doc`."""
    counter = Counter([token.text.lower() for token in doc if token.is_alpha])
    num_hapaxes = len([word for word, count in counter.items() if count == 1])

    return num_hapaxes


def word_count(doc: Doc) -> int:
    """Return the total word count fo `doc`."""
    return len(doc)


def stopword_count(doc: Doc) -> int:
    """Return the number of stop words in `doc`."""
    return sum([token.is_stop for token in doc])


def entity_count(doc: Doc) -> int:
    """Return the number of entities in `doc`."""
    return len(doc.ents)


def upper_count(doc: Doc) -> int:
    """Return the count of uppercase tokens in `doc` excluding`I'."""
    return sum((token.text.isupper() and token.text != "I" for token in doc))


def excessive_punctuation_count(text: str) -> int:
    """Return the count of excessive punctuation marks in `text`."""
    exclamation_points = len(
        re.findall(r"!{2,}", text)
    )  # Two or more exclamation points
    question_marks = len(re.findall(r"\?{2,}", text))  # Two or more ?
    ellipses = len(
        re.findall(r"\.{2,}", text)
    )  # Two or more periods in a row (e.g., "...")
    return exclamation_points + question_marks + ellipses


def repeated_characters_count(text: str) -> int:
    """Return the number of repeated characters e.g. in sooooo good
    in `text`."""
    repeated_chars = len(
        re.findall(r"(.)\1{2,}", text)
    )  # Three or more repeated characters
    return repeated_chars


def repeated_words_count(text: str) -> int:
    """Return the number of repeated words occurrences e.g. so so so
    good in `text`.
    """
    repeated_words = len(
        re.findall(r"\b(\w+)\b\s+\1", text.lower())
    )  # Detect words repeated consecutively
    return repeated_words


def get_sent_score(sentence):
    return TextBlob(sentence).sentiment.polarity


def get_sentiment(doc: Doc) -> tuple[float, float, float, float]:
    """
    Return the average, max, min, and std of sentiment score polarities using
    textblob for each sentence in `doc`.
    """
    # Segment into sentences
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) <= 1:
        return get_sent_score(doc), get_sent_score(doc), get_sent_score(doc), 0

    sentiment_scores = [get_sent_score(sentence) for sentence in sentences]
    average_sentiment = np.mean(sentiment_scores)
    max_sentiment = np.max(sentiment_scores)
    min_sentiment = np.min(sentiment_scores)
    std_sentiment = np.std(sentiment_scores)

    return (average_sentiment, max_sentiment, min_sentiment, std_sentiment)


# TODO: readability scores, keyword presence, text complexity, BOW,
# TFIDF, text complexity, num characters total, avg word length
