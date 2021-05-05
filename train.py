from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import json
import torch
import os
from tqdm import tqdm

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler)


def load_training_data(data_file):
    examples = []
    with open(data_file) as fin:
        count = 0
        for line in fin:
            _ex = json.loads(line)
            if len(_ex) > 0:
                examples.append(torch.tensor(_ex, dtype=torch.long))

            count += 1

    print("Loaded {} examples. ".format(count))
    return examples


def load_model_and_tokenizer(model_name_or_path: str):
    _config = AutoConfig.from_pretrained(model_name_or_path, num_labels=2)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=_config)
    _tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return _model, _tokenizer


def split_minibatch(input_ids, batch_size):
    assert batch_size >= 2, "Batch size needs at least be 2, as each mini batch should include at least 1 positive and negative example"

    _minibatches = []

    pos_example = input_ids[0:1]
    neg_examples = input_ids[1:]
    for i in range(0, neg_examples.size()[0], batch_size - 1):
        _minibatches.append(torch.cat((pos_example, neg_examples[i:i + batch_size - 1]), 0))

    return _minibatches


def train(model,
          tokenizer,
          train_data,
          save_dir,
          lr: float = 1e-5,
          warmup: float = 0.1,
          num_epoch: int = 3,
          batch_size: int = 2,
          limit_example: int = 4,
          save_every_k_step: int = 50000,
          cuda=True,
          **kwargs):

    # First check save_dir, if not empty, throw warning
    assert os.path.exists(save_dir), "save_dir does not exist!"
    assert len(os.listdir(save_dir)) == 0, "save_dir must be empty!"

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=1)

    optimizer = AdamW(model.parameters(),
                      lr=lr,
                      correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup * len(train_dataloader),
                                                num_training_steps=len(train_dataloader) * 10 / batch_size) # TODO: Figure out a better way

    global_step = 0
    model.zero_grad()
    model.train()

    # TODO: allow setting device
    if cuda:
        model.to("cuda")

    for epc in range(num_epoch):
        print("Epoch #{}: \n".format(epc))
        epoch_iterator = tqdm(train_dataloader, desc="Training Steps")

        for step, input_ids in enumerate(epoch_iterator):
            input_ids = input_ids.squeeze(0)

            # When there's no negative example in the current batch, skip the batch
            if input_ids.size()[0] <= 1:
                continue

            if input_ids.size()[0] > limit_example:
                input_ids = input_ids[:limit_example]

            if cuda:
                input_ids = input_ids.cuda()

            minibatches = split_minibatch(input_ids, batch_size=batch_size)
            minibatch_losses = []
            for mini_batch in minibatches:
                outputs = model(mini_batch)
                output_logits = outputs[0]

                # The first instance in each batch is always the positive example
                # The rest of the instances are negatives
                neg_prob = output_logits[1:, 1]
                pos_prob = output_logits[0, 1].repeat(neg_prob.size()[0])

                # We want pos_prob to always rank higher than neg_prob
                # https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss
                rank_lbl = torch.tensor([1] * neg_prob.size()[0], dtype=torch.long)

                # Also force the classification of entailment to be good
                entailment_lbl = torch.zeros(output_logits.size()[0], dtype=torch.long)
                entailment_lbl[0] = 1 # The first example is always positive
                if cuda:
                    rank_lbl = rank_lbl.to("cuda")
                    entailment_lbl = entailment_lbl.to("cuda")

                rank_loss_f = torch.nn.MarginRankingLoss(margin=0)  # Default margin to 0
                entailment_loss_f = torch.nn.CrossEntropyLoss()

                rank_loss = rank_loss_f(pos_prob, neg_prob, rank_lbl)
                entailment_loss = entailment_loss_f(output_logits, entailment_lbl)

                loss = rank_loss + entailment_loss
                loss.backward()

                minibatch_losses.append(loss.item())
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            epoch_iterator.set_description("Loss: {:.3f}".format(sum(minibatch_losses) / len(minibatch_losses)))

            global_step += 1

            if global_step % save_every_k_step == 0:
                save_dir_name = "step_{}".format(global_step)
                save_sub_dir = os.path.join(save_dir, save_dir_name)
                os.mkdir(save_sub_dir)
                model.save_pretrained(save_sub_dir)
                tokenizer.save_pretrained(save_sub_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training script for the entailment model')
    parser.add_argument('model_name_or_path', type=str, help="Name or path of the huggingface model/checkpoint to use")
    parser.add_argument('train_data_file', type=str, help="Path pre-tokenized data")
    parser.add_argument('save_dir', type=str, help="Directory to save the model")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    train_data = load_training_data(args.train_data_file)
    train(model, tokenizer, train_data, args.save_dir)