import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

import metrics
from models.user_config import UserModel

def print_sample(i, generation, truth, metric, score):
    print(f"Sample {i}, generation: {generation}")
    print(f"Sample {i}, truth: {truth}")
    if isinstance(score, tuple) and len(score) == 3:
        print(f"Metric ({metric}): tp {score[0]}, fp {score[1]}, fn {score[2]}")
    else:
        print(f"Metric ({metric}): {score}")
    print()

def run_and_evaluate(data_df, max_eval_rows, print_interval=200):
    model = UserModel()

    if max_eval_rows < len(data_df):
        data_df_eval = data_df.sample(max_eval_rows)
    else:
        data_df_eval = data_df

    # Run model
    outputs = []
    task_methods = {
        'multiple-choice': model.task_multichoice,
        'generation': model.task_generation,
        'retrieval': model.task_retrieval,
        'ranking': model.task_ranking,
        'named_entity_recognition': model.task_named_entity_recognition,
    }

    for _, row in tqdm(data_df_eval.iterrows(), total=len(data_df_eval), desc='Processing'):
        task_type = row['task_type']
        if task_type not in task_methods:
            raise NotImplementedError(f"No task method for {task_type=}")
        
        task_prompt = row['input_field']
        task_fn = task_methods[task_type]
        task_output = task_fn(task_prompt)
        outputs.append(task_output)

    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentence_all_lm = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    sentece_multilingual = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)

    eval_methods = {
        'accuracy': metrics.accuracy,
        'hit rate@3': metrics.hit_rate_3,
        'rougel': metrics.rougel,
        'sent-transformer': lambda g,t: metrics.sent_transformer(g, t, sentence_all_lm),
        'multilingual-sent-transformer': lambda g,t: metrics.sent_transformer(g, t, sentece_multilingual),
        'micro f1': metrics.tp_fp_fn,
        'ndcg': metrics.ndcg_eval,
        'bleu': metrics.bleu,
        'jp-bleu': lambda g,t: metrics.bleu(g,t, jp=True)
    }

    per_task_metrics = {}

    for ri, row in tqdm(data_df_eval.iterrows(), total=len(data_df_eval), desc='Evaluating'):
        metric = row['metric']
        if metric not in eval_methods:
            raise NotImplementedError(f"No metric for {metric=}")

        task_name = row['task_name']
        per_task_metrics.setdefault(task_name, {
            'metric': metric,
            'sample_score': []
        })
        
        gt = row['output_field']
        model_output = outputs[ri]

        eval_fn = eval_methods[metric]
        metric_score = eval_fn(model_output, gt)
        per_task_metrics[task_name]['sample_score'].append(metric_score)
        per_task_metrics[task_name]['sample_score'].append(metric_score)
        
        if ri % print_interval == 0:
            print_sample(ri, model_output, gt, metric, metric_score)

    # Aggregate scores
    for k in per_task_metrics:
        if per_task_metrics[k]['metric'] != 'micro f1':
            print(k, len(per_task_metrics[k]['sample_score']))
            per_task_metrics[k]['overall_metric'] = np.mean(per_task_metrics[k]['sample_score'])
        else:
            per_task_metrics[k]['overall_metric'] = metrics.compute_f1_score(per_task_metrics[k]['sample_score'])

    overall_metrics = {
        'task_name': [],
        'metric': [],
        'overall_score': []
    }
    for k in per_task_metrics:
        overall_metrics['task_name'].append(k)
        overall_metrics['metric'].append(per_task_metrics[k]['metric'])
        overall_metrics['overall_score'].append(per_task_metrics[k]['overall_metric'])
    track_wise_score = np.mean(overall_metrics['overall_score'])
    overall_metrics['task_name'].append('track_wise')
    overall_metrics['metric'].append('track_wise')
    overall_metrics['overall_score'].append(track_wise_score)
    overall_metrics_df = pd.DataFrame(overall_metrics)
    overall_metrics_df.to_json("scores.json", orient='records', lines=True)
    print(f"Overall score {track_wise_score}")

if __name__ == "__main__":
    DATA_FILENAME = './data/tracks/track3_rephrase.json'
    data_df = pd.read_json(DATA_FILENAME, lines=True)
    MAX_EVAL_ROWS = 100000
    run_and_evaluate(data_df, MAX_EVAL_ROWS)