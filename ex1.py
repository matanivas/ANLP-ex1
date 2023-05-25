import numpy as np
from transformers import TrainingArguments, HfArgumentParser, AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification, set_seed, EvalPrediction, Trainer, TextClassificationPipeline
import argparse
from datasets import load_dataset
import pandas as pd
# import wandb
import logging
import timeit
from evaluate import load
from dataclasses import dataclass, field




def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('num_seeds',
                        type=int,
                        nargs='?',
                        default=3)
    parser.add_argument('num_train_samples',
                        type=int,
                        nargs='?',
                        default=-1)
    parser.add_argument('num_valid_samples',
                        type=int,
                        nargs='?',
                        default=-1)
    parser.add_argument('num_test_samples',
                        type=int,
                        nargs='?',
                        default=-1)

    return parser.parse_args()


def compute_metrics(p: EvalPrediction):
    if isinstance(p.predictions, tuple):
      predictions = p.predictions[0]
    else:
      predictions = p.predictions
    predictions = np.argmax(predictions, axis=1)
    met = metric.compute(predictions=predictions, references=p.label_ids)
    return met

def preprocess(data):
    return tokenizer(data['sentence'], truncation=True, max_length=512)

if __name__ == '__main__':
    models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

    parser = HfArgumentParser(TrainingArguments)
    my_args = parse_arguments()
    num_of_seeds = my_args.num_seeds
    num_of_train = my_args.num_train_samples
    num_of_valid = my_args.num_valid_samples
    num_of_test = my_args.num_test_samples
    raw_dataset = load_dataset('sst2')

    training_args = TrainingArguments(output_dir='output')

    results = pd.DataFrame(columns=['model_name', 'seed', 'model', 'accuracy'])
    results.index = results['model_name']
    mean_std = dict()
    tr_start_time = timeit.default_timer()
    for model_name in models:
        metric = load('accuracy')
        training_args.save_strategy = 'no'

        for seed in range(num_of_seeds):
            set_seed(seed)
            training_args.run_name = f'{model_name}_seed_{seed}'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            dataset = raw_dataset.map(preprocess, batched=True)

            if num_of_valid != -1:
              valid_set = dataset['validation'].select(list(range(num_of_valid)))
            else:
              valid_set = dataset['validation']
            
            if num_of_train != -1:
              train_dataset = dataset['train'].select(list(range(num_of_train)))
            else:
              train_dataset = dataset['train']

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_set,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
            )

            trainer.train()
            metrics = trainer.evaluate()

            model_data = {'model_name': model_name,
                          'seed': seed,
                          'model': trainer.model,
                          'accuracy': metrics['eval_accuracy']}

            results = pd.concat([results, pd.DataFrame([model_data])], ignore_index=True)
        results.index = results['model_name']
        model_accuracy = results.loc[model_name]['accuracy']
        mean_std[model_name] = model_accuracy.mean(), model_accuracy.std()

    tr_end_time = timeit.default_timer()


    tr_time = tr_end_time - tr_start_time



    best_model_name = max(mean_std, key=mean_std.get)
    accuracy = results.loc[best_model_name]['accuracy']

    best_seed = 0
    if my_args.num_seeds > 1:
      best_seed = np.argmax(accuracy.to_numpy())
    results = results.set_index(['seed'], append=True)
    best_model = results.loc[(best_model_name, best_seed)]['model']
    text_classifier = TextClassificationPipeline(model=best_model.to('cpu'), tokenizer=tokenizer)
    if my_args.num_test_samples != -1:
      test_set = dataset['test'].select(list(range(my_args.num_test_samples)))
    else:
      test_set = dataset['test']
    pr_start_time = timeit.default_timer()
    outputs = text_classifier(test_set['sentence'], batch_size=1)
    pr_end_time = timeit.default_timer()
    pr_time = pr_end_time - pr_start_time

    with open('predictions.txt', 'w') as f:
        for i in range(len(outputs)):
            f.write(f'{test_set["sentence"][i]}###{outputs[i]["label"][-1]}\n')


    with open('res.txt', 'w') as f:
        for key, value in mean_std.items():
            f.write(f'{key},{value[0]} +- {value[1]}\n')
        f.write('----\n')
        f.write(f'train time,{tr_time}\n')
        f.write(f'predict time,{pr_time}\n')


    results.to_csv('results.csv')
