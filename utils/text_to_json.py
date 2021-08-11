from unicodedata import normalize
import json
import os
import argparse


def get_list(filename: str) -> list:
    text = open(filename, mode="r")
    text_list = text.read().splitlines()
    text_list = list(map(lambda x: normalize("NFKC", x), text_list))
    text.close()
    return text_list


def convert_to_dict(
    source_file: str, target_file: str, source_lang: str, target_lang: str
) -> list:
    source_list = get_list(source_file)
    target_list = get_list(target_file)

    translation_list = []

    for count, line in enumerate(source_list):
        main_dict = {}
        dict_sentence = {}
        dict_sentence[source_lang] = line
        dict_sentence[target_lang] = target_list[count]
        main_dict["translation"] = dict_sentence
        translation_list.append(main_dict)

    return translation_list


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

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    source_path = os.path.join(script_dir, args.source_path)
    target_path = os.path.join(script_dir, args.target_path)
    source_lang = args.source_lang
    target_lang = args.target_lang
    output_file = args.output_file

    print(source_path, target_path)

    translation_list = convert_to_dict(
        source_path, target_path, source_lang, target_lang
    )

    json_dict = {"data": translation_list}

    output_file = open(output_file, "w", encoding="utf8")
    json.dump(json_dict, output_file, indent=4, sort_keys=False, ensure_ascii=False)
    output_file.close()


if __name__ == "__main__":
    main()
