This code implements a BART and a RoBERTa Encoder-Decoder model for doing transfer learning experiments based on [(Zoph et al., 2016)](https://arxiv.org/abs/1604.02201). The idea is to train a model on a high-resource language pair and then transfer some of the learned parameters by freezing the encoder/decoder depending on the child language pairs to be trained. 


## Requirements

The code is written entirely in Python and it requires:

- Pytorch
- SacreBLEU 
- [transformers >= 4.7.0](https://huggingface.co/transformers/)
- [datasets](https://github.com/huggingface/datasets)
- [tokenizers](https://github.com/huggingface/tokenizers)
- Jupyter 
- numpy

## Usage

### Data

The data can be parallel datasets that you have. It is also possible to use the data from huggingface such as WMT14 En-De and load it from the python script. 

### Training

Since the script is designed to load the data in a JSON format (for your local data), assuming you have your parallel sentences in `txt` format. You can use the helper script in utils: `text_to_json.py` to convert sentence pairs in separate txt files into one json file which can then be loaded by the training script. 

If you do not have a dev set, you can use `split_text_files.py` to split your data into train and dev set and only then convert it to a json format. 

Tokenizers for the languages also need to be trained before training the translation model. This is done using `train_tokenizers_fast.py`. This trains a huggingface tokenizer which will then be loaded from the training script. You can specify how many subwords (since we are using BPE) you want in the dict (the norm seems to be 32000). However, you can change it if GPU memory is an issue during training 

The script used to train a translation model from scratch for the parent model is `train_bart.py`. Parameters for the training such as model parameters or training parameters are set in the `.yaml` file: `train_bart_config.yaml`. After training the parent model, we load the parent model by specifying the model directory in `train_bart_from_pretrained.yaml`. Training parameters can also be set at that file. It is possible to change how many encoder/decoder layers have their parameters frozen in the notebook `train_bart_from_pretrained.ipynb`.

During training, the model will evaluate the model at fixed intervals (steps or epochs) which can be set in the yaml file. It will output not only the loss on the dev set but also the BLEU score for the dev set. This is calculated using SacreBLEU.

To run using GPU, just do `CUDA_VISIBLE_DEVICES=X` depending on how many GPUs you want to use. Without that, the script will automatically use all the available GPUs available to it. 

## Acknowledgements

The code is based on the example provided by huggingface on finetuning a pretrained model for [machine translation](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)
