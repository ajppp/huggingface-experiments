import json
import os
from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizerFast, BartTokenizerFast
from transformers import BartConfig, BartModel, BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import torch
import numpy as np
import yaml
import argparse

def main() -> None:

    parser = argparse.ArgumentParser(description="train a BART model from scratch")

    parser.add_argument('config_path',
                        metavar='config_path',
                        type=str,
                        help='the path to the config YAML file')

    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_datasets = load_dataset('json', data_files ={'train': config['train_data_dir'], 'valid': config['valid_data_dir']}, field='data')
    metric = load_metric(config['metric'])

    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['source_tokenizer'])
    id_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config['target_tokenizer'])
    en_tokenizer.unk_token,en_tokenizer.cls_token, en_tokenizer.sep_token,en_tokenizer.pad_token, en_tokenizer.mask_token = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
    id_tokenizer.unk_token, id_tokenizer.cls_token, id_tokenizer.sep_token, id_tokenizer.pad_token, id_tokenizer.mask_token = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']

    encoder_max_length = config['encoder_max_length']
    decoder_max_length = config['decoder_max_length']
    source_lang = config['source_lang']
    target_lang = config['target_lang']

    def process_data_to_model_inputs(batch):
      # tokenize the inputs and labels
      inputs = [ex[source_lang] for ex in batch["translation"]]
      targets = [ex[target_lang] for ex in batch["translation"]]
      inputs = en_tokenizer(inputs, padding="max_length", truncation=True, max_length=encoder_max_length)
      outputs = id_tokenizer(targets, padding="max_length", truncation=True, max_length=decoder_max_length)

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
                                          batch_size=2048,
                                          remove_columns=['translation'])

    bart_config = BartConfig(vocab_size=config['vocab_size'],
                             dropout=config['dropout'],
                             attention_dropout=config['attention_dropout'],
                             encoder_layers=config['encoder_layers'],
                             decoder_layers=config['decoder_layers'],
                             encoder_attention_heads=config['encoder_attention_heads'],
                             decoder_attention_heads=config['decoder_attention_heads'],
                             activation_function=config['activation_function'],
                             max_position_embeddings=config['max_position_embeddings'],
                             encoder_ffn_dim=config['encoder_ffn_dim'],
                             decoder_ffn_dim=config['decoder_ffn_dim'],)
    model = BartForConditionalGeneration(config=bart_config)


    model.config.decoder_start_token_id = en_tokenizer.cls_token_id
    model.config.eos_token_id = en_tokenizer.sep_token_id
    model.config.pad_token_id = en_tokenizer.pad_token_id


    data_collator = DataCollatorForSeq2Seq(tokenizer=en_tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        evaluation_strategy=config['evaluation_strategy'],
        eval_steps=config['eval_steps'],
        learning_rate=float(config['learning_rate']),
        warmup_steps=config['warmup_steps'],
        label_smoothing_factor=config['label_smoothing_factor'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],
        num_train_epochs=config['num_train_epochs'],
        lr_scheduler_type=config['lr_scheduler_type'],
        seed=config['seed'],
        fp16=config['fp16'],
        save_total_limit=config['save_total_limit'],
        save_strategy=config['save_strategy'],
        adafactor=config['adafactor'],
        predict_with_generate=config['predict_with_generate'],
    )


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = id_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = id_tokenizer.pad_token_id
        label_str = id_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # postprocessing
        pred_str, label_str = postprocess_text(pred_str, label_str)

        result = metric.compute(predictions=pred_str, references=label_str)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != id_tokenizer.pad_token_id) for pred in pred_ids]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=en_tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    main()
