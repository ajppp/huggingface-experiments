import json
import os
import argparse
import random


def get_list(filename: str) -> list:
    text = open(filename, mode="r")
    text_list = text.read().splitlines()
    text_list = list(map(lambda x: normalize("NFKC", x), text_list))
    text.close()
    return text_list


def convert_to_dict(
    source_file: str,
    target_file: str,
    source_lang: str,
    target_lang: str,
    valid_percentage: int,
) -> tuple:
    source_list = get_list(source_file)
    target_list = get_list(target_file)

    n = len(source_list)
    num_valid = int(valid_percentage * n / 100)
    print(n, num_valid)
    valid_indices = random.sample(range(n), num_valid)
    source_valid_list = []
    target_valid_list = []
    train_translation_list = []
    valid_translation_list = []

    j = 0
    for i in valid_indices:
        source_valid_list.append(source_list.pop(i - j))
        target_valid_list.append(target_list.pop(i - j))
        j += 1

    for count, line in enumerate(source_list):
        main_dict = {}
        dict_sentence = {}
        dict_sentence[source_lang] = line
        dict_sentence[target_lang] = target_list[count]
        main_dict["translation"] = dict_sentence
        train_translation_list.append(main_dict)

    for count, line in enumerate(source_valid_list):
        main_dict = {}
        dict_sentence = {}
        dict_sentence[source_lang] = line
        dict_sentence[target_lang] = target_valid_list[count]
        main_dict["translation"] = dict_sentence
        valid_translation_list.append(main_dict)

    return train_translation_list, valid_translation_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="converts sentence pairs stored in txt files to a json format to permit easy loading by huggingface datasets module"
    )

    parser.add_argument(
        "source_path",
        metavar="source_path",
        type=str,
        help="the path to the source language data",
    )

    parser.add_argument(
        "target_path",
        metavar="target_path",
        type=str,
        help="the path to the target language data",
    )

    parser.add_argument(
        "source_lang", metavar="source_lang", type=str, help="the source language"
    )

    parser.add_argument(
        "target_lang", metavar="target_lang", type=str, help="the target language"
    )

    parser.add_argument(
        "output_file",
        metavar="output_file",
        type=str,
        help="the path to the output file",
    )

    parser.add_argument(
        "percentage",
        metavar="percentage",
        type=int,
        help="the percentage of the data that should go to the dev set",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    source_path = os.path.join(script_dir, args.source_path)
    target_path = os.path.join(script_dir, args.target_path)
    source_lang = args.source_lang
    target_lang = args.target_lang
    output_file = args.output_file
    percentage = args.percentage

    train_output_file: str = output_file + "_train.json"
    valid_output_file: str = output_file + "_valid.json"

    print(source_path, target_path)

    train_translation_list, valid_translation_list = convert_to_dict(
        source_path, target_path, source_lang, target_lang, percentage
    )

    train_json_dict = {"data": train_translation_list}
    valid_json_dict = {"data": valid_translation_list}

    train_output_file = open(train_output_file, "w", encoding="utf8")
    valid_output_file = open(valid_output_file, "w", encoding="utf8")
    json.dump(
        train_json_dict,
        train_output_file,
        indent=4,
        sort_keys=False,
        ensure_ascii=False,
    )
    json.dump(
        valid_json_dict,
        valid_output_file,
        indent=4,
        sort_keys=False,
        ensure_ascii=False,
    )
    train_output_file.close()
    valid_output_file.close()


if __name__ == "__main__":
    main()
