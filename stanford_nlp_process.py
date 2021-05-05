import stanza
import json
from typing import List, Tuple


def tokenize_sentence(text: str) -> List[Tuple[str, int, int]]:
    """
    A better sentence tokenizer than the ones in spacy and nltk
    :param text:
    :return: A list of sentences, along with the character offsets of the sentence boundaries
    """
    prev_char = None
    sent_start = 0
    text_len = len(text)
    sentences = []
    for char_idx, char in enumerate(text):
        if char == '.':
            if prev_char and (not prev_char.isupper()):
                if (char_idx + 1 >= text_len) or (not text[char_idx + 1].islower()):
                    sentences.append((text[sent_start: char_idx + 1], sent_start, char_idx + 1))
                    sent_start = char_idx + 1
        prev_char = char

    if len(sentences) == 0:
        sentences.append((text, 0, len(text)))

    return sentences


def valid_line(line: str) -> bool:
    """
    Input text that is too long will break the LSTM models in stanza, so we skip inputs that are too long...
    :param line:
    :return:
    """
    _words = line.split(" ")
    return len(_words) < 2000


def process_dataset_stanza(dataset_file: str,
                           output_path: str) -> None:
    nlp = stanza.Pipeline('en')
    with open(dataset_file) as fin, open(output_path, 'w') as fout:
        count = 0
        for line in fin:
            line = line.strip()

            if not valid_line(line):
                doc_dict = []
            else:
                try:
                    doc = nlp(line)
                    doc_dict = doc.to_dict()
                except Exception as e:
                    print(e)
                    doc_dict = []

            fout.write(json.dumps(doc_dict))
            fout.write('\n')

            count += 1

            if count % 500 == 0:
                print("Processed: {}".format(count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process a line separated dataset with Stanza (Stanford CoreNLP)')
    parser.add_argument('dataset_file', type=str, help="Path to a line separated dataset file")
    parser.add_argument('output_path', type=str, help="Output path")
    args = parser.parse_args()

    process_dataset_stanza(args.dataset_file, args.output_path)