from feqa import FEQA
import benepar
import nltk
import json


def download_models():
        benepar.download('benepar_en2')
        nltk.download('stopwords')


def load_lines(fname):
        with open(fname) as fin:
                lines = fin.readlines()
                lines = [l.strip() for l in lines]

        return lines


def evaluate(source_file, summary_file, result_file):
        docs = load_lines(source_file)
        sums = load_lines(summary_file)
        
        scorer = FEQA(use_gpu=True)

        score = scorer.compute_score(docs, sums, aggregate=False)
        score = [float(s) for s in score]

        with open(result_file, 'w') as fout:
            json.dump(score, fout)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run FEQA on a list of source and summary text.')
    parser.add_argument('source_file', type=str, help="Path to a newline-separated file containing the source articles/text.")
    parser.add_argument('summary_file', type=str, help="Path to a newline-separated file containing the summaries")
    parser.add_argument('result_file', type=str, help="where to save the results (in .json)")
    args = parser.parse_args()

    download_models()
    evaluate(args.source_file, args.summary_file, args.result_file)
