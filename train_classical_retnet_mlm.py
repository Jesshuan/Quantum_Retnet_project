from dataclasses import dataclass

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset, concatenate_datasets

import torch

import pdb

import pickle

import re

from retnet.modeling_retnet import RetNetForCausalLM
from retnet.configuration_retnet import load_config_from_json

torch.cuda.empty_cache()


MODEL_STORE_PATH = "./model_store/model_small_mlm/"

@dataclass
class MyArgs:
    model_size: str = '300m'
    text_col: str = 'text'
    max_length: int = 128


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()


    try:
        with open("./dataset_store/train_dataset", "rb") as f:
            train_dataset = pickle.load(f)

        with open("./dataset_store/eval_dataset", "rb") as f:
            eval_dataset = pickle.load(f)

        print("Datasets train/eval imported...")

    except:

        # First split

        dataset_1_original= load_dataset("bookcorpus", split="train")

        # First split - 10% => 7,4M rows
        dataset_1_original = dataset_1_original.train_test_split(test_size=0.1)["test"]

        dataset_1_original = dataset_1_original.remove_columns([col for col in dataset_1_original.column_names if col!="text"])

        # Second split - Train/Test split (test 1% = 74k rows)
        dataset_1_original = dataset_1_original.train_test_split(test_size=0.01)

        train_dataset = dataset_1_original["train"]
        eval_dataset = dataset_1_original["test"]

        with open("./dataset_store/train_dataset", "wb") as f:
            pickle.dump(train_dataset, f)

        with open("./dataset_store/eval_dataset", "wb") as f:
            pickle.dump(eval_dataset, f)

        print("Datasets train/eval re-builded...")

    try:
        model = RetNetForCausalLM.from_pretrained("./model_store/model_small_mlm")

        print('Model imported from a saved file...')
    except:
        config = load_config_from_json(f"configs/retnet-{args.model_size}/config.json")
        model = RetNetForCausalLM(config)

        print('Model instanciated...')

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = 128
    tokenizer.pad_token = tokenizer.eos_token

    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def transform(sentence):

        return re.sub('[^A-Za-z0-9.,;!?]+', ' ', sentence) + tokenizer.eos_token

    def tokenize_datset(example):
        example = [transform(sentence) for sentence in example[args.text_col]]
        input_ids = tokenizer(example,
                              truncation=True,
                              padding=True,
                              max_length=128,
                              return_tensors='pt').input_ids
        return {'input_ids': input_ids}
    
    train_dataset = train_dataset.shuffle().map(tokenize_datset, remove_columns=train_dataset.column_names, num_proc=6, batched=True, batch_size=16)
    eval_dataset = eval_dataset.shuffle().map(tokenize_datset, remove_columns=eval_dataset.column_names,num_proc=6, batched=True, batch_size=16)


    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if train_args.do_train:
        trainer.train()
        trainer.save_model(output_dir=MODEL_STORE_PATH)
        print("Model saved.")
    #if train_args.do_eval:
        #trainer.evaluate()
        #print("Model evaluated.")


if __name__ == "__main__":
    main()
