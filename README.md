# Entity & Quantity Correction for Abstractive Summarization

This repository contains the code and resources to ["Improving Faithfulness in Abstractive Summarization
with Contrast Candidate Generation and Selection"](https://www.seas.upenn.edu/~sihaoc/static/pdf/CZSR21.pdf) in NAACL'21.  
```
@inproceedings{CZSR21,
    author = {Sihao Chen and Fan Zhang and Kazoo Sone and Dan Roth},
    title = {{Improving Faithfulness in Abstractive Summarization with Contrast Candidate Generation and Selection}},
    booktitle = {NAACL},
    year = {2021}
}
```

## Use the Correction Model for Summary Ranking
The trained BART-base model for classifying whether a summary is hallucinated/faithful is published to huggingface model hub as [`CogComp/bart-faithful-summary-detector`](https://huggingface.co/CogComp/bart-faithful-summary-detector). With the `transformers` library installed, you can use it as follows.  

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("CogComp/bart-faithful-summary-detector")
model = AutoModelForSequenceClassification.from_pretrained("CogComp/bart-faithful-summary-detector")

article = "Ban-Ki Moon was re-elected for a second term by the UN General Assembly, unopposed and unanimously, on 21 June 2011"

bad_summary = "Ban Ki-moon was elected for a second term in 2007"
good_summary = "Ban Ki-moon was elected for a second term in 2011"

bad_pair = tokenizer(text=bad_summary, text_pair=article, return_tensors='pt')
good_pair = tokenizer(text=good_summary, text_pair=article, return_tensors='pt')

bad_score = model(**bad_pair)
good_score = model(**good_pair)

print(good_score[0][:, 1] > bad_score[0][:, 1]) # True, label mapping: "0" -> "Hallucinated" "1" -> "Faithful"
```

## Example Corrections and Evaluation 
We include the 1,510 examples in XSum test set that our method made corrections to under `data/`. 

- `source.part.txt`: source text/article
- `target.part.txt`: ground truth summary
- `bart.part.txt`: summaries generated by the [BART (large)](https://huggingface.co/facebook/bart-large-xsum) baseline
- `corrected.part.txt`: summaries corrected by our system 

To reproduce the evaluation results:

- `ROUGE`: We use version `0.0.4` of [`rouge-score`](https://pypi.org/project/rouge-score/) library. 
- `BertScore`: We use version `0.3.6` of [bert-score](https://github.com/Tiiiger/bert_score), with the `roberta-large_L17_no-idf_version=0.3.6` model. See their [github readme]((https://github.com/Tiiiger/bert_score)) for instructions. 
- `FEQA`: See the example usage below for `run_feqa.py`. Check the [FEQA](https://github.com/esdurmus/feqa/blob/master/feqa.py) repo for the complete list of required libraries. Note: You may want to use a fresh environment for FEQA, as it requires a different version of `transformers`. 

```bash
python run_feqa.py \  
    --source_file data/source.part.txt \
    --summary_file data/corrected.part.txt \ 
    --result_file data/feqa_corrected_results.json
```
## Create Unfaithful Variants via Entity Perturbation
Please install the following packages:
```
stanza
word2number # We use this to normalize surface forms of quantities and numbers  
```

We use [`stanza`](https://stanfordnlp.github.io/stanza/) to extract the named entities in text. For exact reproducibility, please install `stanza=1.1.1`.
```
pip install stanza
```
Download the english models with the following python snippet:
```python
import stanza
stanza.download('en') # download English model
```
Put the source text and summary text in two line-separated files respectively (See `data/source.part.txt` and `data/target.part.txt` for example). 
First annotate the two files with NER by running
```
python stanford_nlp_process.py source.txt source.stanza
python stanford_nlp_process.py target.txt target.stanza
```   
This will create `source.stanza` and `target.stanza` as two jsonline files; each json would be the annotated version of 
source and target text.

Next, generate alternative versions of the summary by running:
```
python make_entity_perturbations.py \
    --source_stanza_output source.stanza \
    --source_file source.txt \
    --target_stanza_output target.stanza \ 
    --target_file target.txt \
    --output_path  train.jsonl
```
This will generate alternative summaries in **training mode**, i.e. only generate alternative versions if 
all entities in the original summary have appeared in the source document. This is to make sure that we can safely
use the original summary as the "positive" examples during training. 

In **evaluation mode**, it's the other way around -- we only want to generate variants when the original summary is 
hallucinated. To run the script in evaluation mode, add the `--eval` flag.
```
python make_entity_perturbations.py \
    --source_stanza_output source.stanza \
    --source_file source.txt \
    --target_stanza_output target.stanza \ 
    --target_file target.txt \
    --output_path  test.jsonl
    --eval
```
You can also control the maximum number of variants generated for each instance. e.g. `--limit=10`. 

  
## Training 
The training data and validation data we generated (by following the steps outlined in the previous section) can be
 downloaded from this [google drive folder](https://drive.google.com/drive/folders/18Eqfemxf6wOQeSUNrZMlacF2OvwaRQ87?usp=sharing).

Please install the following packages:
```
transformers==3.4.0
tqdm
torch
```
With the `transformers` installed (we used `transformers==3.4.0`), first run `bpe_tokenize.py` on each of the 
train/val split to cache tokenized input. For example, 
```
python bpe_tokenize.py \
    --model_name facebook/bart-base \
    --data_file train.jsonl \
    --output_path train.tokenized 
```
Run the training script. By default `cuda` is enabled.  
```
python train.py \
    --model_name facebook/bart-base \ 
    --train_data_file train.tokenized \
    --save_dir model/
```