import argparse
import os
import pandas as pd
import json


def raw_data_load(args):
    with open(os.path.join(args.raw_data_dir, args.raw_data_file), "rb") as f:
        f = json.load(f)

    df = pd.DataFrame(f["data"])
    return df


def make_dict(df):
    paragraphs_dict = {"instruction": [], "input": [], "output": []}
    for i in range(df.shape[0]):
        paragraphs_dict["instruction"].append(df.iloc[i, 0][0]["qas"][0]["question"])
        paragraphs_dict["input"].append(df.iloc[i, 0][0]["context"])
        paragraphs_dict["output"].append(df.iloc[i, 0][0]["qas"][0]["answers"][0]["text"])
    return paragraphs_dict


def add_data(df, data_dict):
    df["instruction"] = data_dict["instruction"]
    df["input"] = data_dict["input"]
    df["output"] = data_dict["output"]
    df.drop(["paragraphs", "title"], axis=1, inplace=True)
    return df


def save_to_json(args, df):
    df.to_json(os.path.join(args.save_data_path, args.save_file_name), force_ascii=False, orient="records")



def main(args):
    df = raw_data_load(args)
    data_dict = make_dict(df)
    df = add_data(df, data_dict)

    save_to_json(args, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="/home/yehoon/workspace/data/ai_hub/")
    parser.add_argument("--raw_data_file", type=str, default="ko_wiki_v1_squad.json")
    parser.add_argument("--save_data_path", type=str, default=os.path.join(os.curdir, "data"))
    parser.add_argument("--save_file_name", type=str, default="ai_hub.json")

    args = parser.parse_args()

    main(args)
