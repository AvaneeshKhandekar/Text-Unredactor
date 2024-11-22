import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import logging
import en_core_web_sm
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
spacy_model = en_core_web_sm.load()

sia = SentimentIntensityAnalyzer()


def load_and_preprocess_data(file1, file2):
    df1 = pd.read_csv(file1, sep="\t", names=["split", "name", "context"], on_bad_lines="skip")
    df2 = pd.read_csv(file2, sep="\t", names=["split", "name", "context"], on_bad_lines="skip")
    df = pd.concat([df1, df2]).reset_index(drop=True)
    df["context"] = df["context"].str.replace(r"█+", "[MASK]", regex=False)
    return df


def split_data(df):
    train_df = df[df["split"] == "training"]
    valid_df = df[df["split"] == "validation"]
    return train_df, valid_df


def fit_vectorizers(train_sentences):
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
    count_vectorizer.fit(train_sentences)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
    tfidf_vectorizer.fit(train_sentences)

    return count_vectorizer, tfidf_vectorizer


def extract_features(row, count_vectorizer, tfidf_vectorizer):
    sentence = row["context"]
    redacted_word = row["name"]

    features = {
        **{f"ngram_{i}": val for i, val in enumerate(count_vectorizer.transform([sentence]).toarray()[0])},
        **{f"tfidf_{i}": val for i, val in enumerate(tfidf_vectorizer.transform([sentence]).toarray()[0])},
        "redacted_length": len(redacted_word),
        "redacted_spaces": redacted_word.count(" "),
        **sia.polarity_scores(sentence),
        "num_capitals": sum(c.isupper() for c in sentence),
        "sentence_length": len(word_tokenize(sentence)),
    }

    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    try:
        idx = words.index("[MASK]")
        features["prev_word"] = words[idx - 1] if idx > 0 else ""
        features["next_word"] = words[idx + 1] if idx < len(words) - 1 else ""
        features["prev_pos"] = pos_tags[idx - 1][1] if idx > 0 else ""
        features["next_pos"] = pos_tags[idx + 1][1] if idx < len(words) - 1 else ""
    except ValueError:
        features.update({"prev_word": "", "next_word": "", "prev_pos": "", "next_pos": ""})

    doc = spacy_model(sentence)
    entity_counts = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
    features.update(entity_counts)

    features["num_sentences"] = len(list(doc.sents))

    features["semantic_similarity"] = \
    cosine_similarity([spacy_model(sentence).vector], [spacy_model(redacted_word).vector])[0][0]

    return features


def transform_features(df, count_vectorizer, tfidf_vectorizer, dict_vectorizer):
    features = df.apply(lambda row: extract_features(row, count_vectorizer, tfidf_vectorizer), axis=1).tolist()
    return dict_vectorizer.transform(features)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(y_valid, predictions, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']

    logging.info(f"Precision: {precision:.5f}")
    logging.info(f"Recall: {recall:.5f}")
    logging.info(f"F1-Score: {f1_score:.5f}")


def save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer):
    if not os.path.exists("results"):
        os.makedirs("results")

    if model:
        joblib.dump(model, "results/model.pkl")
    if count_vectorizer:
        joblib.dump(count_vectorizer, "results/count_vectorizer.pkl")
    if tfidf_vectorizer:
        joblib.dump(tfidf_vectorizer, "results/tfidf_vectorizer.pkl")
    if dict_vectorizer:
        joblib.dump(dict_vectorizer, "results/dict_vectorizer.pkl")


def test_submission(model, dict_vectorizer, count_vectorizer, tfidf_vectorizer, test_file, output_file):
    test_df = pd.read_csv(test_file, sep="\t", names=["id", "context"], on_bad_lines="skip")

    logging.info("Preprocessing test data.")
    test_df["context"] = test_df["context"].str.replace(r"█+", "[MASK]", regex=False)

    logging.info("Extracting features for test data.")

    test_features = test_df.apply(
        lambda row: extract_features(
            {"context": row["context"], "name": "[MASK]"},
            count_vectorizer,
            tfidf_vectorizer,
        ),
        axis=1,
    ).tolist()

    X_test = dict_vectorizer.transform(test_features)

    logging.info("Making predictions on test data.")
    predictions = model.predict(X_test)

    submission_df = pd.DataFrame({"id": test_df["id"], "name": predictions})

    submission_df.to_csv(output_file, sep="\t", index=False, header=False)
    logging.info(f"Submission file created: {output_file}")


def main():
    df = load_and_preprocess_data("data/unredactor.tsv", "data/redacted_output.tsv")
    train_df, valid_df = split_data(df)

    if os.path.exists("results/count_vectorizer.pkl") and os.path.exists("results/tfidf_vectorizer.pkl"):
        logging.info("Loading existing vectorizers.")
        count_vectorizer = joblib.load("results/count_vectorizer.pkl")
        tfidf_vectorizer = joblib.load("results/tfidf_vectorizer.pkl")
    else:
        logging.info("Fitting new vectorizers.")
        count_vectorizer, tfidf_vectorizer = fit_vectorizers(train_df["context"])
        save_artifacts(None, count_vectorizer, tfidf_vectorizer, None)

    logging.info("Extracting and transforming features.")
    dict_vectorizer = DictVectorizer(sparse=False)
    X_train = dict_vectorizer.fit_transform(
        train_df.apply(lambda row: extract_features(row, count_vectorizer, tfidf_vectorizer), axis=1).tolist())
    X_valid = transform_features(valid_df, count_vectorizer, tfidf_vectorizer, dict_vectorizer)
    y_train, y_valid = train_df["name"], valid_df["name"]

    if os.path.exists("results/model.pkl"):
        logging.info("Loading existing model.")
        model = joblib.load("results/model.pkl")
    else:
        logging.info("Training new model.")
        model = train_model(X_train, y_train)
        save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer)

    logging.info("Evaluating model.")
    evaluate_model(model, X_valid, y_valid)

    logging.info("Generating submission file.")
    test_submission(model, dict_vectorizer, count_vectorizer, tfidf_vectorizer, "data/test.tsv", "submission.tsv")


if __name__ == "__main__":
    main()
