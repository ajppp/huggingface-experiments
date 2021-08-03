import json
import os
from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizerFast
from transformers import RobertaConfig, RobertaModel, RobertaForCausalLM, EncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

raw_datasets = load_dataset('json', data_files={'train': 'en_pt_es_data/train_en_pt.json', 'valid': 'en_pt_es_data/valid_en_pt.json'}, field='data')
metric = load_metric("sacrebleu")

en_tokenizer = PreTrainedTokenizerFast(tokenizer_file="en_pt_tokenizers/en_test.json")
es_tokenizer = PreTrainedTokenizerFast(tokenizer_file="en_pt_tokenizers/pt_test.json")
en_tokenizer.unk_token,en_tokenizer.cls_token, en_tokenizer.sep_token,en_tokenizer.pad_token, en_tokenizer.mask_token = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
es_tokenizer.unk_token, es_tokenizer.cls_token, es_tokenizer.sep_token, es_tokenizer.pad_token, es_tokenizer.mask_token = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']

encoder_max_length = 256
decoder_max_length = 256
source_lang = 'en'
target_lang = 'pt'

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = [ex[source_lang] for ex in batch["translation"]]
  targets = [ex[target_lang] for ex in batch["translation"]]
  inputs = en_tokenizer(inputs, padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = es_tokenizer(targets, padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == en_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch


tokenized_datasets = raw_datasets.map(process_data_to_model_inputs,
                                      batched=True,
                                      batch_size=256,
                                      remove_columns=['translation'])


tokenized_datasets.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

print(tokenized_datasets['train']['labels'])

vocabsize = 30000
encoder_config = RobertaConfig(vocab_size = vocabsize,
                    max_position_embeddings = 1024,
                    num_attention_heads = 8,
                    num_hidden_layers = 8,
                    hidden_size = 768,
                    hidden_dropout_prob=0.2,
                    attention_probs_dropout_prob=0.1)
encoder = RobertaModel(config=encoder_config)
decoder_config = encoder_config
decoder_config.add_cross_attention = True
decoder_config.is_decoder = True
decoder = RobertaForCausalLM(config=decoder_config)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.decoder_start_token_id = en_tokenizer.cls_token_id
model.config.eos_token_id = en_tokenizer.sep_token_id
model.config.pad_token_id = en_tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size



model.to(device)

batch_size = 28
args = Seq2SeqTrainingArguments(
    output_dir="./roberta_pt_checkpoints",
    evaluation_strategy="steps",
    eval_steps=2000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # learning_rate=6e-4,
    weight_decay=1e-5,
    max_grad_norm=1.0,
    warmup_steps=4000,
    num_train_epochs=50,
    # label_smoothing_factor=0.1,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=10,
    adafactor=True
)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = es_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = es_tokenizer.pad_token_id
    label_str = es_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # postprocessing
    pred_str, label_str = postprocess_text(pred_str, label_str)

    result = metric.compute(predictions=pred_str, references=label_str)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != es_tokenizer.pad_token_id) for pred in pred_ids]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=en_tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()












