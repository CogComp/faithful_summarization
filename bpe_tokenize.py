from transformers import AutoTokenizer
import json


def tokenize_data(data_file, model_name, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(data_file) as fin, open(output_path, 'w') as fout:
        count = 0
        for json_str in fin:
            example = json.loads(json_str)

            _tgt_batch = []
            _src_batch = []
            # only consider the first positive
            if len(example["positive_examples"]) == 0:
                continue

            # TODO: we assume only 1 positive examples exist for every instance
            _tgt_batch.append(example["positive_examples"][0])
            _tgt_batch += example["negative_examples"]
            
            _src_batch = [example["source_text"]] * len(_tgt_batch)
            tokenized = tokenizer(text=_tgt_batch,
                                  text_pair=_src_batch,
                                  padding=True,
                                  max_length=tokenizer.max_len,
                                  truncation=True)

            fout.write(json.dumps(tokenized["input_ids"]))
            fout.write("\n")

            count += 1

            if count % 500 == 0:
                print("Processed: {}".format(count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run (model specific) BPE tokenization on training data, and cache the tokenization results')
    parser.add_argument('model_name', type=str, help="Name of the huggingface model to use")
    parser.add_argument('data_file', type=str, help="Path to train/test/eval data file")
    parser.add_argument('output_path', type=str, help="Path to output file")
    args = parser.parse_args()

    tokenize_data(args.data_file, args.model_name, args.output_path)
