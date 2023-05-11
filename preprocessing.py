import os
import json
import glob
import gzip
import shutil
from tqdm import tqdm
import pickle
#
# import nest_asyncio
# nest_asyncio.apply()

import pandas as pd
from deepl import DeepLCLI
import googletrans



# # Unzip json file
# unzip_data_path = glob.glob("../data/c4/realnewslike/*.gz")
# save_dir = "./data/c4"
# for gz_file in unzip_data_path:
#     with gzip.open(gz_file, 'rb') as f_in:
#         file_name = (".").join(gz_file.split("/")[-1].split(".")[:-1])
#         with open(os.path.join(save_dir,file_name), 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)
#
#
# # Merge Files
# train_files = glob.glob("data/c4/*train*.json")
# valid_files = glob.glob("data/c4/*valid*.json")[0]
#
# val_df = pd.read_json(valid_files, lines=True)
#
# train_df = pd.DataFrame()
# for file in tqdm(train_files, desc="json merging..."):
#     tmp = pd.read_json(file, lines=True)
#     train_df = pd.concat([train_df, tmp], axis=0)
#
#
# # Save Files as pickle file
# with open("data/train_total_realnewslike.pkl", "wb") as f:
#     pickle.dump(train_df, f)
#     f.close()
#
# with open("data/valid_total_realnewslike.pkl", "wb") as f:
#     pickle.dump(val_df, f)
#     f.close()


# Translate with DeepL
print("Load .pkl data...")

with open("data/train_total_realnewslike.pkl", "rb") as f:
    train_df = pickle.load(f)
    f.close()

with open("data/valid_total_realnewslike.pkl", "rb") as f:
    valid_df = pickle.load(f)
    f.close()

print("Loading is Done..!")


# Translate with DeepL
def deepl_translate(df):
    deepl = DeepLCLI("en", "ko")
    # deepl.timeout = 15000
    # deepl.max_length = 3000
    kor = []
    for idx in tqdm(range(len(df)), desc="Translate ... "):
        if 3000<len(df.iloc[idx,0])<=6000:
            tmp = df.iloc[idx,0].split("\n")

            front = (" ").join(tmp[:len(tmp)//2])
            back = (" ").join(tmp[len(tmp)//2:])

            front_tr = deepl.translate(front)
            back_tr = deepl.translate(back)

            kor.append((front_tr+back_tr).replace("\n", " "))

        elif len(df.iloc[idx,0])>6000:
            continue
        else:
            tr = deepl.translate(df.iloc[idx,0])

            kor.append(tr.replace("\n", " "))

    return kor


# Translate with googletrans
def google_translate(df):
    translator = googletrans.Translator()
    kor = []

    for idx in tqdm(range(len(df)), desc="Translate ... "):
        kor.append(translator.translate(df.iloc[idx, 0], src="en", dest="ko").text.replace("\n", " "))

    return kor


train_df["kor"] = google_translate(train_df)
valid_df["kor"] = google_translate(valid_df)
