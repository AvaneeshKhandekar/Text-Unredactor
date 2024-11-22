import pytest
from unittest.mock import MagicMock, patch
from redactor import init_model, redact_names, redact_file, main


@pytest.fixture
def get_nlp():
    nlp = init_model()
    return nlp


def test_redact_names(get_nlp):
    nlp = get_nlp
    text = "Hello, my name is John Doe."
    doc = nlp(text)
    redaction_char = "█"
    censored_terms = []
    tsv_rows = []

    redact_names(text, doc, redaction_char, censored_terms, tsv_rows)

    assert len(censored_terms) == 1
    assert censored_terms[0]['term'] == "John Doe"
    assert censored_terms[0]['start'] == 18
    assert censored_terms[0]['end'] == 26
    assert censored_terms[0]['type'] == "PERSON"
    assert len(tsv_rows) == 1
    assert "John Doe" in tsv_rows[0]


def test_redact_file(get_nlp):
    mock_nlp = get_nlp
    mock_file_obj = MagicMock()
    mock_file_obj.read.return_value = b"Hello, my name is John."

    tsv_rows = []
    censored_terms = redact_file(mock_file_obj, mock_nlp, tsv_rows)

    assert len(censored_terms) == 1
    assert censored_terms[0]["term"] == "John"


@patch('re.sub')
@patch('tarfile.open')
@patch('random.sample')
@patch('en_core_web_sm.load')
def test_main(mock_spacy_load, mock_random_sample, mock_tarfile_open, mock_re_sub):
    mock_spacy_load.return_value = MagicMock()

    mock_random_sample.return_value = ['file1.txt', 'file2.txt']

    mock_tar = MagicMock()
    mock_tarfile_open.return_value = mock_tar

    mock_file1 = MagicMock()
    mock_file1.read.return_value = b"Hello, my name is John Doe."
    mock_file2 = MagicMock()
    mock_file2.read.return_value = b"Jane Doe went to the market."

    def extractfile_side_effect(filename):
        if filename == 'file1.txt':
            return mock_file1
        elif filename == 'file2.txt':
            return mock_file2
        return None

    mock_tar.extractfile.side_effect = extractfile_side_effect

    mock_re_sub.side_effect = lambda pattern, repl, string: string.replace(pattern, repl)

    with patch('sys.argv', ['script.py', '--output', '/path/to/output', '--samples', '2']):
        main()

    mock_spacy_load.assert_called()
    mock_tarfile_open.assert_called()


def test_redact_names_no_entities():
    text = "There are no names to redact here."
    doc_mock = MagicMock()
    doc_mock.ents = []
    redaction_char = "█"
    censored_terms = []
    tsv_rows = []

    redact_names(text, doc_mock, redaction_char, censored_terms, tsv_rows)

    assert len(censored_terms) == 0
    assert len(tsv_rows) == 0
