from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import os
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="trains tokenizers for sentences stored in txt format and save the tokenizer as a JSON"
    )

    parser.add_argument(
        "data_path", metavar="data_path", type=str, help="the path to the language data"
    )

    parser.add_argument("lang", metavar="lang", type=str, help="the language")

    parser.add_argument(
        "output_directory",
        metavar="output_directory",
        type=str,
        help="the path to the output directory",
    )

    parser.add_argument(
        "vocab_size",
        metavar="vocab_size",
        type=int,
        help="the number of subwords in the dict",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    source_path = os.path.join(script_dir, args.source_path)
    source_lang = args.source_lang
    output_directory = args.output_directory
    vocab_size = args.vocab_size
    saved_file = output_directory + "/" + source_lang + "_tok.json"

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # initialize the two tokenizers and the trainer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    # train tokenizer
    files = [source_path]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files, trainer)

    tokenizer.save(saved_file)

    print("Tokenizer JSON file saved at:", os.path.relpath(saved_file))


if __name__ == "__main__":
    main()
