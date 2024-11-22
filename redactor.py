import argparse
import os
import re
import sys
import tarfile
import random
import logging

import nltk
import en_core_web_sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def init_model():
    logging.info("Initializing models...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nlp = en_core_web_sm.load()
    logging.info("Models initialized.")
    return nlp


def redact_names(text, doc, redaction_char, censored_terms, tsv_rows):
    original_text = text
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            censored_terms.append({
                "term": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "type": "PERSON"
            })
            entity = re.sub(r'^.*?>"', '', ent.text)
            redacted_text = original_text.replace(entity, redaction_char * len(ent.text))
            tsv_line = f"training\t{entity}\t{redacted_text}\n"
            if tsv_line not in tsv_rows:
                tsv_rows.append(tsv_line)


def redact_file(file_obj, nlp, tsv_rows):
    text = file_obj.read().decode('utf-8')
    text = re.sub(r'/>"', '"', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('¬', '')
    censored_terms = []
    doc = nlp(text)
    redact_names(text, doc, '█', censored_terms, tsv_rows)
    return censored_terms


def main():
    parser = argparse.ArgumentParser(description='Text Redactor')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--stats', help='Output redaction stats file path')
    parser.add_argument('--samples', type=int, default=500, help='Number of data to process')
    args = parser.parse_args()

    logging.info("Starting redaction process...")
    nlp = init_model()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Created output directory: {args.output}")

    tar_path = 'data/imdb_data.tar'
    with tarfile.open(tar_path, 'r') as tar:
        extracted_files = [file for file in tar.getnames() if file.endswith('.txt')]

    logging.info(f"Found {len(extracted_files)} data in archive.")
    selected_files = random.sample(extracted_files, min(args.samples, len(extracted_files)))
    logging.info(f"Selected {len(selected_files)} data.")

    tsv_output_file = os.path.join(args.output, 'redacted_output.tsv')
    tsv_rows = []
    stats = []

    with tarfile.open(tar_path, 'r') as tar:
        for input_file in selected_files:
            logging.info(f"Processing file: {input_file}")
            file_obj = tar.extractfile(input_file)
            censored_terms = redact_file(file_obj, nlp, tsv_rows)

            if censored_terms:
                term_counts = {}
                for term in censored_terms:
                    term_type = term["term"].lower()
                    term_counts[term_type] = term_counts.get(term_type, 0) + 1

                stats.append(f"Processed file: {input_file}\n")
                stats.append(f"Censored Terms Count: {len(censored_terms)}\n")
                for term, count in term_counts.items():
                    stats.append(f"Term: {term}, Count: {count}\n")
                    for censored in censored_terms:
                        if censored["term"].lower() == term:
                            stats.append(
                                f"  - Start: {censored['start']}, End: {censored['end']}, Type: {censored['type']}\n")

    if tsv_rows:
        logging.info(f"Writing redacted terms to {tsv_output_file}.")
        with open(tsv_output_file, 'w', encoding='utf-8') as tsv_file:
            tsv_file.writelines(tsv_rows)

    if args.stats:
        logging.info("Writing stats...")
        if args.stats.lower() == 'stdout':
            sys.stdout.writelines(stats)
        elif args.stats.lower() == 'stderr':
            sys.stderr.writelines(stats)
        else:
            with open(args.stats, 'w', encoding='utf-8') as stats_file:
                stats_file.writelines(stats)

    logging.info("Redaction process completed.")


if __name__ == "__main__":
    main()
