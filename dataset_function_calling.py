import json
import os

import numpy as np
from datasets import Dataset


def filter_test_train_jsons(data_dir):
    file_list = os.listdir(data_dir)
    file_list = [f for f in file_list if f.endswith(".json")]
    train_files = [f for f in file_list if "train" in f]
    test_files = [f for f in file_list if "test" in f]
    train_files = [os.path.join(data_dir, f) for f in train_files]
    test_files = [os.path.join(data_dir, f) for f in test_files]
    return train_files, test_files


def create_dataset(files):
    data = []
    for file in files:
        with open(file) as f:
            data += json.load(f)

    text_data = [json.dumps(entry) for entry in data]
    dataset = Dataset.from_dict({"text": text_data})
    return dataset


def create_train_test_datasets(data_dir):
    train_files, test_files = filter_test_train_jsons(data_dir)
    train_dataset = create_dataset(train_files)
    test_dataset = create_dataset(test_files)
    return train_dataset, test_dataset


def process_dataset(dataset, enc, input_len=1024):
    data = json.loads(dataset["text"])
    system_prompt = "system: " + data["system"] + "\n"
    user_prompt = "user: " + data["user"] + "\n"
    response = "assistant: " + data["assistant"]

    prompt = system_prompt + user_prompt
    prompt_ids = enc.encode_ordinary(prompt)
    prompt_id_len = len(prompt_ids)
    prompt_ids.append(enc.eop_token)

    response_ids = enc.encode_ordinary(response)
    response_ids.append(enc.eot_token)
    prompt_ids = prompt_ids + response_ids
    prompt_response_len = len(prompt_ids)

    prompt_ids = prompt_ids + [enc.pad_token] * (input_len - len(prompt_ids) + 1)
    prompt_ids = np.array(prompt_ids, dtype=np.uint16)

    prompt_mask = np.array([1] * prompt_id_len + [0] * (input_len - prompt_id_len))
    prompt_mask = np.array(prompt_mask, dtype=np.uint8)

    pad_mask = np.array([0] * input_len)
    pad_mask[prompt_response_len - 1 :] = 1
    pad_mask = np.array(pad_mask, dtype=np.uint8)

    out = {
        "output_ids": prompt_ids,
        "length": prompt_response_len,
        "prompt_mask": prompt_mask,
        "pad_mask": pad_mask,
    }

    return out


def get_datasets(data_dir, enc):
    train_dataset, test_dataset = create_train_test_datasets(data_dir)
    train_dataset = train_dataset.map(lambda x: process_dataset(x, enc))
    test_dataset = test_dataset.map(lambda x: process_dataset(x, enc))
    train_dataset = train_dataset.filter(lambda x: x["length"] <= 1024)
    test_dataset = test_dataset.filter(lambda x: x["length"] <= 1024)
    return train_dataset, test_dataset
