from dataclasses import dataclass

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorWithPadding)
from datasets import load_dataset

import torch

import pickle

import re


from quantum_retnet.modeling_quantum_retnet import RetNetForSequenceClassification
from quantum_retnet.configuration_quantum_retnet import load_config_from_json

torch.cuda.empty_cache()

MODEL_STORE_PATH = "./model_store/model_small_quantum_classifier/"

TRAIN_DATASET_PATH = "./dataset_store/train_dataset_for_classifier"

TEST_DATASET_PATH = "./dataset_store/test_dataset_for_classifier"

TOKEN_ENV_PATH = "./.env"

BACKEND = None


@dataclass
class MyArgs:
    model_size: str = '300m'
    dataset_name: str = 'sst2'
    text_col: str = 'sentence'
    label_col: str = 'label'
    max_length: int = 30
    #backend: str="ibm_brisbane"

def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))

    train_args, args = parser.parse_args_into_dataclasses()

    # Train / Eval dataset

    try:
        with open(TRAIN_DATASET_PATH, "rb") as f:
            train_dataset = pickle.load(f)

        with open(TEST_DATASET_PATH, "rb") as f:
            eval_dataset = pickle.load(f)

        print("Datasets train/eval imported...")

    except:

        # Split

        train_dataset = load_dataset(args.dataset_name, split="train")
        eval_dataset = load_dataset(args.dataset_name, split="validation")

        with open(TRAIN_DATASET_PATH, "wb") as f:
            pickle.dump(train_dataset, f)

        with open(TEST_DATASET_PATH, "wb") as f:
            pickle.dump(eval_dataset, f)

        print("Datasets train/eval re-builded...")

    try:
        model = RetNetForSequenceClassification.from_pretrained(MODEL_STORE_PATH)

        print('Model imported from a saved file...')
    except:
        #config = load_config_from_json(f"configs/{args.model_size}/config.json", backend=args.backend, token_env_path=TOKEN_ENV_PATH)
        #config = load_config_from_json(f"configs/{args.model_size}/config.json")
        config = load_config_from_json(f"configs/{args.model_size}/config.json", backend=BACKEND)
        model = RetNetForSequenceClassification(config)

        print('Model instanciated...')

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    #tokenizer.model_max_length = 12
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.unk_token = tokenizer.eos_token
    #tokenizer.bos_token = tokenizer.eos_token

    def transform(sentence):
        return re.sub('[^A-Za-z0-9.,;!?]+', ' ', sentence) + tokenizer.eos_token

    def tokenize_datset(example):
        #example[args.text_col] = [transform(sentence) for sentence in example[args.text_col]]
        example[args.text_col] = transform(example[args.text_col])
        input_ids = tokenizer(example[args.text_col],
                              truncation=True,
                              padding='max_length',
                              max_length=args.max_length,
                              return_tensors='pt')
        #label = example[args.label_col]
        return {"input_ids":input_ids["input_ids"][0], "attention_mask":input_ids["attention_mask"][0]}

    train_dataset = train_dataset.map(tokenize_datset, remove_columns=['idx', 'sentence'])
    eval_dataset = eval_dataset.map(tokenize_datset, remove_columns=['idx', 'sentence'])

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer))

    if train_args.do_train:
            trainer.train()
            trainer.save_model(output_dir=MODEL_STORE_PATH)
    #if train_args.do_eval:
        #trainer.evaluate()


if __name__ == "__main__":
    main()
