# Text Unredactor

## AUTHOR

Avaneesh Khandekar

## INSTALLATION

To install the required dependencies:

```bash
pipenv install
```

## USAGE

To generate additional samples for training:

```bash
pipenv run python redactor.py --output <output_file_path> --samples <number_of_samples> --stats <stats_file_path> 
```

Required:

``` 
- <output> -> Example: /files to create a folder for the additional samples.
```

Optional:

```
- --samples -> Number of additional samples to generate from a subset of IMDB movie review data (default is 500).
- --stats -> Create a stats file containing redaction information
```

To train extract features from the redacted data and train a model:

```bash
pipenv run python unredactor.py 
```

System Specification:
```
CPU: Intel Core i7-9750H @ 2.6 GHz
GPU: NVIDIA GeForce RTX 2060 6 GB
RAM: 16 GB
```
## OVERVIEW

This script is designed to reverse the redaction of sensitive information in text documents.
It uses machine learning and traditional feature extraction techniques to predict and recover redacted terms
based on the patterns found in the text.

It focuses on extracting various features from sentences, including n-grams, sentiment, capitalization, part-of-speech
tagging, and named entity recognition, to train a machine learning model capable of predicting the original redacted
word.

The unredaction script uses ```SpaCy``` for Named Entity Recognition (NER), ```NLTK``` for part-of-speech tagging, and
```VADER``` Sentiment Analysis for sentiment scoring. It uses both CountVectorizer and TF-IDF Vectorizer to capture
important textual features that help predict the redacted word.

The redactor script is used to generate additional samples for training to combine with the provided unredactor.tsv
dataset.

### Work Flow: Unredactor

Feature Extraction: For each sentence, a variety of features are extracted:

- Ngram Features: Unigrams and bigrams are extracted using both ```CountVectorizer``` and ```TF-IDF``` Vectorizer.
    - Captures word patterns and relationships
    - Important for understanding the sentence structure and identifying potential context around redacted terms.
    - ```TF-IDF``` Assigns importance to rare words in the dataset.
    - Helps highlight unique textual characteristics that may hint at the redacted term.
- Sentiment Analysis: Polarity scores (positive, negative, neutral, and compound) are calculated using
  ```SentimentIntensityAnalyzer```.
    - Captures the emotional tone of the sentence, which may relate to the redacted term (e.g., positive names in
      compliments).
- Redacted length and Spaces: The length of the redacted word and the number of spaces in the redacted word.
    - Useful for identifying multi-word redactions or names based on structure.
- Capitalization: The number of capitalized letters in the sentence.
    - Indicates the complexity or verbosity of the sentence.
- Sentence Length: The length of the sentence in terms of word count.
    - Indicates the complexity or verbosity of the sentence.
- POS Tagging: The previous and next words, along with their POS tags, are extracted.
    - Crucial for understanding grammatical and semantic context, aiding in redaction reversal.
- Named Entity Recognition (NER): Named entity labels (e.g., PERSON, DATE, LOCATION, etc.), with each entity type having
  a value of 1 if present in the sentence using ```SpaCy```.
    - Identifies whether the sentence mentions key entities, narrowing down redaction possibilities.
- Dependency Parsing: The number of dependency relations in the sentence.
    - Reflects sentence complexity and structural relationships, providing additional contextual signals.

These features collectively improve the feature vector with lexical, semantic, syntactic, and contextual information,
enabling the model to predict the redacted term with higher accuracy.

Model Training:\
These extracted features are used to train a Random Forest model to predict the original redacted word. The model is
trained on the redacted sentences from the training data.

Prediction:\
The trained model is then used to predict the original name from sentences with redacted names.

### Functions unredactor.py

#### - `load_and_preprocess_data(file1, file2):`

- Loads the dataset from two input files and preprocesses it.
- Reads two files using pd.read_csv, assigning column names ("split", "redacted", "sentence").
- Concatenates both datasets into a single DataFrame and resets the index.
- Replaces any instances of "█" characters in the sentences with the token "[REDACTED]".
- Returns the concatenated and preprocessed DataFrame.

#### - `split_data(df):`

- Splits the DataFrame into training and validation datasets.
- Filters the DataFrame based on the "Split" column, where rows with "training" are separated into train_df and those
  with "validation" are separated into valid_df.
- Returns the training (train_df) and validation (valid_df) DataFrames.

#### - `fit_vectorizers(train_sentences):`

- Fits the CountVectorizer and TfidfVectorizer to the training sentences.
- CountVectorizer is fitted on unigrams and bigrams (ngram_range=(1, 2)) and excludes English stop words.
- TfidfVectorizer is similarly fitted but applies TF-IDF weighting.
- Returns the trained CountVectorizer and TfidfVectorizer.

#### - `extract_features(row, count_vectorizer, tfidf_vectorizer):`

- This function extracts various features from a sentence:
- Ngram features using both CountVectorizer and TF-IDF Vectorizer.
- Uses VADER Sentiment Analysis to calculate sentiment polarity scores for the sentence.
- Counts Redacted word length and number of spaces to add to features.
- Counts the number of capitalized letters and the sentence length (in terms of words).
- Extracts the previous and next words and their respective POS tags based on the tokenized sentence.
- Uses apcy to extract named entities and add to features.
- Uses SpaCy to identify syntactic dependencies in the sentence and add to features.
- Returns a dictionary of features.

#### - `transform_features(df, count_vectorizer, tfidf_vectorizer, dict_vectorizer):`

- Transforms the entire DataFrame of features into a format suitable for training.
- Applies extract_features for each row in the DataFrame to generate a list of feature dictionaries.
- Uses DictVectorizer to transform these feature dictionaries into a sparse matrix.
- Returns the transformed features as a sparse matrix.

#### - `train_model(X_train, y_train):`

- Trains a machine learning model (Random Forest classifier).
- A RandomForestClassifier is instantiated and trained on the extracted features (X_train) and the corresponding target
  labels (y_train).
- Returns the trained Random Forest model.

#### - `evaluate_model(model, X_valid, y_valid):`

- Evaluates the performance of the trained model.
- Uses the trained model to make predictions on the validation dataset (X_valid).
- Calculates the accuracy of the predictions.
- Logs the precision, recall, and f1-score using classification_report.

#### - `save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer):`

- Saves the trained model and vectorizers to disk for later use.
- Checks if a directory named results exists; if not, creates it.
- Saves the model and vectorizers using joblib to the results folder.

#### - `save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer):`

- Saves the trained model and vectorizers to disk for later use.
- Checks if a directory named results exists; if not, creates it.
- Saves the model and vectorizers using joblib to the results folder.

#### - `main():`

- Main pipeline that integrates all the functions.
- Loads and preprocesses the data from the specified input files.
- Splits the data into training and validation datasets.
- Checks if pre-existing vectorizers and models are available, loading them if so.
- If not, fits new vectorizers using the training sentences.
- Extracts features and transforms them using DictVectorizer.
- Loads an existing model or trains a new one if necessary.
- Evaluates the trained model on the validation dataset and logs the results.

### Tests unredactor.py

#### - `test_load_and_preprocess_data(mock_data):`

- Function Tested: ```load_and_preprocess_data(file1, file2)```.
- Verifies that data is loaded and concatenated correctly.
- Asserts the shape of the resulting DataFrame matches expectations.

#### - `test_split_data(mock_data):`

- Function Tested: ```split_data(df)```.
- Verifies the data is split into training and validation sets.
- Asserts the correct number of rows in training and validation DataFrames.

#### - `test_fit_vectorizers(mock_data):`

- Function Tested: ```fit_vectorizers(train_sentences)```.
- Verifies the initialization and fitting of CountVectorizer and TfidfVectorizer.
- Asserts that the returned objects are instances of their respective vectorizer classes.

#### - `test_extract_features(mock_data):`

- Function Tested: ```extract_features(row, count_vectorizer, tfidf_vectorizer)```.
- Verifies the correct feature dictionary is generated for a given sentence.
- Asserts that the returned value is a dictionary.

#### - `test_transform_features(mock_data):`

- Function Tested: ```transform_features(df, count_vectorizer, tfidf_vectorizer, dict_vectorizer)```.
- Verifies the transformation of features using vectorizers.
- Asserts that the dict_vectorizer.transform() method is called.

#### - `test_train_model(mock_data):`

- Function Tested: ```train_model(X_train, y_train)```.
- Verifies the training process of the RandomForestClassifier.
- Asserts that the model's fit method is called with the correct arguments.

#### - `test_evaluate_model(capsys):`

- Function Tested: ```evaluate_model(model, X_valid, y_valid)```.
- Verifies the evaluation of the model on validation data.
- Asserts that that accuracy, precision, recall, and F1-score are logged correctly.

#### - `test_redact_phones():`

- Function Tested: ```redact_phones(text, symbol, censored_terms)```.
- Verifies that phone numbers are redacted properly from the text
- Asserts that the phone number is not present in the redacted text and that it is recorded in the censored terms.

#### - `test_save_artifacts():`

- Function Tested: ```save_artifacts(model, count_vectorizer, tfidf_vectorizer, dict_vectorizer)```.
- Verifies the saving model and vectorizers to disk.
- Asserts that joblib.dump is called for each artifact.

#### - `test_main:`

- Function Tested: ```main()```.
- Verifies the complete pipeline from loading data to evaluation.
- Asserts that appropriate functions are called and flow is executed correctly.

### Results

The current model yields low performance metrics:

- Precision: 0.02243
- Recall: 0.02777
- F1-Score: 0.02311
- Accuracy: 4.03%

The model relies on basic feature engineering (e.g., n-grams, POS tags, sentiment) but lacks deep contextual
understanding of language.
It struggles to generalize from limited patterns in the provided dataset.
The training data may not include sufficient diversity or examples, leading to overfitting on training instances and
poor generalization to unseen data.
Predicting names from minimal context is challenging, especially when entities can vary widely in form and structure.

Ideas for improvement:

- Using a large, diverse dataset of redacted and unredacted texts to provide more examples for the model to learn
  patterns.
- Using a pre-trained generative model like BERT, fine-tuned on redacted text data, to leverage its ability to predict
  missing tokens in a given context.
- Generative models can capture nuanced relationships between words and infer redacted terms more accurately.
- Using embeddings to replace traditional vectorizers, providing richer semantic context.

By training on a large, diverse corpus with a generative approach, the model will better understand context and improve
its ability to predict names or entities in unseen data.

### ASSUMPTIONS:

- **Available Context**: Assumes enough contextual information is present in the surrounding text for accurate predictions.
- **Dataset**: Assumes the training and validation datasets are clean, diverse, and accurately labeled.
- **Dataset**: Assumes only one name is redacted per row in the dataset. If the sentence has multiple names, it has multiple rows.
- **Language**: It is assumed that all text will be in standard English Language.

### BUGS:

- **Feature overlap**: Some features (e.g., n-grams and TF-IDF) might duplicate information. However, since the data is small, both are being used Ref: https://stackoverflow.com/questions/27496014/does-it-make-sense-to-use-both-countvectorizer-and-tfidfvectorizer-as-feature-ve)
- **Generalization**: If the training data lacks diverse or sufficient examples, the model fails to generalize to new or unseen contexts.
- **SpaCy**: The spaCy model may miss certain entities reducing redaction effectiveness.

## Supplementary Code for Data Augmentation

### Work Flow: redactor

This script is designed to redact sensitive information, specifically personal names, from text documents.
It uses SpaCy's Named Entity Recognition (NER) model to identify names within text and replaces them with a specified
redaction character (█).

The script processes .txt files within a .tar archive (imdb_data.tar), redacting detected names and storing the results
in a TSV file. It also generates detailed statistics about the redaction process.

#### Stats File

- The stats file summarizes the processing of input files during the redaction operation.
- It includes details such as the file name, total count of censored terms, and for each term, the number of occurrences
  along with their start and end indices in the text.

### Functions redactor.py

#### - `init_model():`

- Initialize and download required NLP models (nltk and SpaCy).
- Downloads nltk datasets and loads SpaCy's English model.
- Returns nlp (SpaCy NLP model).

#### - `redact_names(text, doc, redaction_char, censored_terms, tsv_rows):`

- Redacts names from the text.
- Identifies person entities using SpaCy's named entity recognition (NER).
- Replaces detected names with a redaction character (█).
- Updates censored_terms with redacted details and appends redacted lines to tsv_rows.

#### - `redact_file(file_obj, nlp, tsv_rows):`

- Redacts names from the file.
- Reads and preprocesses text (removes unwanted characters/tags).
- Calls redact_names() for redacting person entities.
- Returns the list of censored_terms.

#### - `main():`

- Entry point of the program, managing the redaction pipeline.
- Calls init_model() for loading models.
- Creates the output directory if it doesn't exist.
- Opens the .tar archive (imdb_data.tar).
- Filters .txt files and randomly selects files based on --samples.
- Iterates through selected files, calls redact_file(), and aggregates statistics.
- Writes redacted terms to a TSV file.
- Outputs redaction statistics to the specified file or stream (stdout, stderr).

### Tests redactor.py

#### - `test_redact_names(get_nlp):`

- Function Tested: ```redact_names(text, doc, redaction_char, censored_terms, tsv_rows)```.
- Verifies that names in the text are properly identified and redacted.
- Asserts the correct number of censored terms is detected, redacted term, its position, and type match expectations,
  TSV row contains the redacted term.

#### - `test_redact_file(get_nlp):`

- Function Tested: ```redact_file(file_obj, nlp, tsv_rows)```.
- Verifies the end-to-end functionality of reading a file, detecting names, and storing redacted terms.
- Asserts the correct number of censored terms is identified.

#### - `test_main(mock_spacy_load, mock_random_sample, mock_tarfile_open, mock_re_sub):`

- Function Tested: ```main()```.
- Mocks all external dependencies to test the main workflow, including file extraction, redaction, and output handling.
- Asserts SpaCy model is loaded, tar archive is opened correctly, Selected files are processed.

#### - `test_redact_names_no_entities():`

- Function Tested: ```redact_names(text, doc, redaction_char, censored_terms, tsv_rows)```.
- Verifies that the function handles cases with no redactable entities gracefully.
- Asserts no terms are detected, no rows are added to the TSV file.

### ASSUMPTIONS:

- **Model Dependency**: Accuracy of identifying entities largely depends on the SpaCy model used.
- **Language**: It is assumed that all text will be in standard English Language.

### BUGS:

- **Name Redaction**: Spacy model missed some names and sometimes redacts non-names like 'Exorcist'.
