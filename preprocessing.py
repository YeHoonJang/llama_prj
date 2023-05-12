import argparse
import os
import json
import glob
import gzip
import shutil
import pickle
import threading
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
#
# import nest_asyncio
# nest_asyncio.apply()

import pandas as pd
from deepl import DeepLCLI
import googletrans


# Unzip json file
def unzip_json_file(args):
    unzip_data_path = glob.glob(os.path.join(args.raw_data_dir, "*.gz"))
    data_dir = os.path.join(args.curr_dir, args.data_dir, args.data_type)
    for gz_file in tqdm(unzip_data_path, desc="Unzipping Data ... "):
        with gzip.open(gz_file, 'rb') as f_in:
            file_name = ".".join(gz_file.split("/")[-1].split(".")[:-1])
            with open(os.path.join(data_dir, file_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    print("Unzipping data is DONE !")


# Merge and Save Files
def merge_and_save_files(args):
    train_files = glob.glob(os.path.join(args.data_dir, args.data_type, "*train*.json"))
    valid_files = glob.glob(os.path.join(args.data_dir, args.data_type, "*valid*.json"))

    val_df = pd.read_json(valid_files[0], lines=True)
    # val_df["kor"] = translate(args, valid_files[0])

    # with open(os.path.join(args.data_dir, "valid_translated_realnewslike.pkl"), "wb") as f:
    #     pickle.dump(val_df, f)
    #     f.close()

    if not os.path.isdir(os.path.join(args.data_dir, "merged")):
        os.mkdir(os.path.join(args.data_dir, "merged"))

    train_df = pd.DataFrame()
    for file in tqdm(train_files, desc="Merging and saving train data ..."):
        tmp = pd.read_json(file, lines=True)
        # tmp["kor"] = translate(args, file)

        train_df = pd.concat([train_df, tmp["text"]], axis=0)
        file_num = file.split("/")[-1].split("-of-")[0]
        if train_df.shape[0] > 1300000:
            with open(os.path.join(args.data_dir, "merged", f"{file_num[3:]}.pkl"), "wb") as f:
                pickle.dump(train_df, f)
                f.close()
            train_df = pd.DataFrame()

    print("Merging and saving train data is DONE !")

    # Save part
    # print("Saving Train Data ... ")
    # with open(os.path.join(args.data_dir, "train_total_realnewslike.pkl"), "wb") as f:
    #     pickle.dump(train_df, f)
    #     f.close()
    # print("Saving Train Data Is DONE !")
    print("Saving validation data ... ")
    file_num = valid_files[0].split("/")[-1].split("-of-")[0]
    with open(os.path.join(args.data_dir, "merged", f"{file_num[3:]}.pkl"), "wb") as f:
        pickle.dump(val_df["text"], f)
        f.close()

    print("Saving validation data is DONE !")

# Translate
# TODO: need to refactoring
def translate(args, json_file):
    # # translate valid
    # if "valid" in json_file:
    #     # valid_file = glob.glob(os.path.join(args.data_dir, args.data_type, json_file))
    #     valid_df = pd.read_json(json_file, lines=True)
    #
    #     if args.translator == "google":
    #         valid_df["kor"] = google_translator(valid_df)
    #     elif args.translator == "deepl":
    #         valid_df["kor"] = deepl_translator(valid_df)
    #
    # # translate train
    # else:
    #     # train_files = glob.glob(os.path.join(args.data_dir, args.data_type, json_file))
    #
    #     # train_df = pd.DataFrame()
    #     # for file in tqdm(train_files, desc="Translating Data ..."):
    #     train_df = pd.read_json(json_file, lines=True)
    #
    #     if args.translator == "google":
    #         train_df["kor"] = google_translator(train_df)
    #     elif args.translator == "deepl":
    #         train_df["kor"] = deepl_translator(train_df)

    # train_df = pd.concat([train_df, tmp], axis=0)

    # print("Translating Data Is DONE !")

    # Save Files as pickle file
    # print("Saving Data ...")

    df = pd.read_json(json_file, lines=True)

    if args.translator == "google":
        df["kor"] = google_translator(df)
    elif args.translator == "deepl":
        df["kor"] = deepl_translator(df)

    return df


def to_txt(args):
    train_pkl = glob.glob(os.path.join(args.data_dir, "merged/train*.pkl"))
    valid_pkl = os.path.join(args.data_dir, "merged/validation.00000.pkl")

    # train
    for i in tqdm(train_pkl, desc="Saving train.pkl to train.txt ... "):
        with open(i, "rb") as f:
            train_df = pickle.load(f)

        f = open(os.path.join(args.data_dir, "train.txt"), 'a', encoding='utf-8')

        for idx in range(len(train_df)):
            f.write(train_df.iloc[idx, 0])
        f.close()
    print("Saving is DONE !")

    # validation
    print("Saving valid.pkl to valid.txt ... ")
    with open(valid_pkl, "rb") as f:
        valid_df = pickle.load(f)

    f = open(os.path.join(args.data_dir, "valid.txt"), 'a', encoding='utf-8')
    for idx in range(len(valid_df)):
        f.write(valid_df.iloc[idx])
    f.close()
    print("Saving is DONE !")


# Load Data
def load_data(args):
    with open(os.path.join(args.data_dir, "train_total_realnewslike.pkl"), "rb") as f:
        train_df = pickle.load(f)
        f.close()

    with open(os.path.join(args.data_dir, "valid_total_realnewslike.pkl"), "rb") as f:
        valid_df = pickle.load(f)
        f.close()

    return train_df, valid_df


# Translate with DeepL
def deepl_translator(df):
    deepl = DeepLCLI("en", "ko")
    # deepl.timeout = 15000
    # deepl.max_length = 3000
    kor = []
    for idx in tqdm(range(len(df)), desc="Translating ... "):
        if 3000 < len(df.iloc[idx, 0]) <= 6000:
            tmp = df.iloc[idx, 0].split("\n")

            front = " ".join(tmp[:len(tmp) // 2])
            back = " ".join(tmp[len(tmp) // 2:])

            front_tr = deepl.translate(front)
            back_tr = deepl.translate(back)

            kor.append((front_tr + back_tr).replace("\n", " "))

        elif len(df.iloc[idx, 0]) > 6000:
            continue
        else:
            tr = deepl.translate(df.iloc[idx, 0])

            kor.append(tr.replace("\n", " "))

    return kor


# Translate with googletrans
def google_translator(df):
    translator = googletrans.Translator()
    kor = []

    for idx in tqdm(range(len(df)), desc="Translating ... "):
        kor.append(translator.translate(df.iloc[idx, 0], src="en", dest="ko").text.replace("\n", " "))

    return kor


def main(args):
    if args.is_unzip:
        unzip_json_file(args)

    if args.is_merge_and_save:
        merge_and_save_files(args)

    if args.is_to_txt:
        to_txt(args)

    # # Load Data
    # print("Load .pkl data...")
    # train_df, valid_df = load_data(args)
    # print("Loading is Done..!")
    # threads = []

    if args.is_translate:
        translated_dir = os.path.join(args.data_dir, "translated")
        if not os.path.isdir(translated_dir):
            os.mkdir(translated_dir)

        files = glob.glob(os.path.join(args.data_dir, args.data_type, "*.json"))

        df = pd.DataFrame()
        for file in tqdm(files, desc="Translating data ... "):
            file_name = file.split("-of-")[0]

            translated = translate(args, file)

            df = pd.concat([df, translated], axis=0)

            if "valid" in file:
                with open(os.path.join(translated_dir, f"{file_name}.pkl"), "wb") as f:
                    pickle.dump(df, f)
                    f.close()

            # save df per 10 json
            if "train" in file and df.shape[0] > 250000:
                with open(os.path.join(translated_dir, f"{file.split('-of-')[0]}.pkl"), "wb") as f:
                    pickle.dump(df, f)
                    f.close()
                df = pd.DataFrame()  # dataframe initialize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="/home/yehoon/workspace/data/c4/realnewslike")
    parser.add_argument("--curr_dir", type=str, default=os.getcwd())
    parser.add_argument("--data_dir", type=str, default="/home/yehoon/workspace/llama_prj/data")
    parser.add_argument("--data_type", type=str, default="c4_realnewslike")

    parser.add_argument("--is_unzip", type=bool, default=False)
    parser.add_argument("--is_merge_and_save", type=bool, default=False)
    parser.add_argument("--is_to_txt", type=bool, default=False)
    parser.add_argument("--is_translate", type=bool, default=False)

    parser.add_argument("--translator", type=str, default=None, choices=['google', 'deepl'])

    args = parser.parse_args()

    main(args)
