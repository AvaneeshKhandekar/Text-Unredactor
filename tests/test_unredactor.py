import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from unredactor import (
    load_and_preprocess_data,
    split_data,
    fit_vectorizers,
    extract_features,
    transform_features,
    train_model,
    evaluate_model,
    save_artifacts,
    main
)


@pytest.fixture
def mock_data():
    data = {
        "Split": ["training", "training", "validation", "validation"],
        "Redacted": ["word1", "word2", "word3", "word4"],
        "Sentence": [
            "This is a test sentence 1.",
            "This is a test sentence 2.",
            "This is a test sentence 3.",
            "This is a test sentence 4.",
        ],
    }
    return pd.DataFrame(data)


def test_load_and_preprocess_data(mock_data):
    with patch("pandas.read_csv", return_value=mock_data) as mock_read_csv:
        df = load_and_preprocess_data("file1.tsv", "file2.tsv")
        assert df.shape == (8, 3)
        mock_read_csv.assert_called()


def test_split_data(mock_data):
    train_df, valid_df = split_data(mock_data)
    assert train_df.shape == (2, 3)
    assert valid_df.shape == (2, 3)


def test_fit_vectorizers(mock_data):
    count_vectorizer, tfidf_vectorizer = fit_vectorizers(mock_data["Sentence"])
    assert isinstance(count_vectorizer, CountVectorizer)
    assert isinstance(tfidf_vectorizer, TfidfVectorizer)


def test_extract_features(mock_data):
    count_vectorizer, tfidf_vectorizer = fit_vectorizers(mock_data["Sentence"])
    row = mock_data.iloc[0]
    features = extract_features(row, count_vectorizer, tfidf_vectorizer)
    assert isinstance(features, dict)


def test_transform_features(mock_data):
    count_vectorizer, tfidf_vectorizer = fit_vectorizers(mock_data["Sentence"])
    dict_vectorizer = MagicMock()
    dict_vectorizer.transform = MagicMock(return_value=[[1, 2], [3, 4]])

    transform_features(mock_data, count_vectorizer, tfidf_vectorizer, dict_vectorizer)
    dict_vectorizer.transform.assert_called()


def test_train_model(mock_data):
    count_vectorizer, tfidf_vectorizer = fit_vectorizers(mock_data["Sentence"])
    dict_vectorizer = MagicMock()
    X_train = dict_vectorizer.fit_transform(
        mock_data.apply(lambda row: extract_features(row, count_vectorizer, tfidf_vectorizer), axis=1).tolist()
    )
    y_train = mock_data["Redacted"]

    with patch("unredactor.RandomForestClassifier") as MockRandomForest:
        mock_model = MagicMock()
        MockRandomForest.return_value = mock_model

        model = train_model(X_train, y_train)

        assert model == mock_model
        mock_model.fit.assert_called_once_with(X_train, y_train)


def test_evaluate_model(capsys):
    model = MagicMock()
    model.predict = MagicMock(return_value=["word1", "word2", "word3", "word4"])

    X_valid = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y_valid = ["word1", "word2", "word3", "word4"]

    with patch("sklearn.metrics.accuracy_score", return_value=1.0), \
            patch("sklearn.metrics.classification_report",
                  return_value={"macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}}):
        evaluate_model(model, X_valid, y_valid)

        captured = capsys.readouterr()
        # assert "Accuracy" in captured.out
        # assert "Precision" in captured.out
        # assert "Recall" in captured.out
        # assert "F1-Score" in captured.out


def test_save_artifacts():
    model = MagicMock()
    count_vectorizer = MagicMock()
    tfidf_vectorizer = MagicMock()
    dict_vectorizer = MagicMock()

    with patch("joblib.dump") as mock_dump:
        save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer)
        mock_dump.assert_called()


@patch("unredactor.load_and_preprocess_data")
@patch("unredactor.split_data")
@patch("unredactor.train_model")
@patch("unredactor.joblib.load")
@patch("unredactor.save_artifacts")
@patch("unredactor.evaluate_model")
def test_main(
        mock_evaluate_model,
        mock_save_artifacts,
        mock_joblib_load,
        mock_train_model,
        mock_split_data,
        mock_load_and_preprocess_data,
        mock_data
):
    mock_load_and_preprocess_data.return_value = mock_data
    mock_split_data.return_value = (mock_data, mock_data)
    mock_joblib_load.side_effect = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

    main()

    mock_train_model.assert_not_called()
    mock_save_artifacts.assert_not_called()
    mock_evaluate_model.assert_called_once()
