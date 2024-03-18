import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

import metrics
import parsers


def print_sample(i, generation, truth, metric, score):
    print(f"Sample {i}, generation: {generation}")
    print(f"Sample {i}, truth: {truth}")
    if isinstance(score, tuple) and len(score) == 3:
        print(
            f"Metric ({metric}): tp {score[0]}, fp {score[1]}, fn {score[2]}"
        )
    else:
        print(f"Metric ({metric}): {score}")
    print()


if __name__ == "__main__":

    # Load Development Data
    DATA_FILENAME = "./data/development.json"
    data_df = pd.read_json(DATA_FILENAME, lines=True)

    # Load UserModel
    from models.user_config import UserModel

    model = UserModel()

    # Generate Responses
    outputs = []
    for _rowd_idx, row in tqdm(
        data_df.iterrows(),
        total=len(data_df),
        desc="Generating Responses",
    ):
        print("=" * 100)
        is_multiple_choice = row["task_type"] == "multiple-choice"
        prompt = row["input_field"]
        model_output = model.predict(prompt, is_multiple_choice)
        outputs.append(model_output)

        print(prompt, model_output)

    # Merge outputs into DF
    data_df["outputs"] = outputs
    print(data_df)

    # Evaluate
    print_interval = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_all_lm = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    sentece_multilingual = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ).to(device)

    eval_methods = {
        "accuracy": metrics.accuracy,
        "hit rate@3": metrics.hit_rate_3,
        "rougel": metrics.rougel,
        "sent-transformer": lambda g, t: metrics.sent_transformer(
            g, t, sentence_all_lm
        ),
        "multilingual-sent-transformer": lambda g, t: metrics.sent_transformer(
            g, t, sentece_multilingual
        ),
        "micro f1": metrics.tp_fp_fn,
        "ndcg": metrics.ndcg_eval,
        "bleu": metrics.bleu,
        "jp-bleu": lambda g, t: metrics.bleu(g, t, jp=True),
    }

    task_parsers = {
        "multiple-choice": parsers.ShoppingBenchTaskParsers("multichoice"),
        "generation": parsers.ShoppingBenchTaskParsers("generation"),
        "retrieval": parsers.ShoppingBenchTaskParsers("retrieval"),
        "ranking": parsers.ShoppingBenchTaskParsers("ranking"),
        "named_entity_recognition": parsers.ShoppingBenchTaskParsers(
            "named_entity_recognition"
        ),
    }

    per_task_metrics = {}

    for row_idx, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Evaluating"
    ):
        metric = row["metric"]
        if metric not in eval_methods:
            raise NotImplementedError(f"No metric for {metric=}")

        task_type = row["task_type"]

        task_name = f"{task_type}---{metric}"
        per_task_metrics.setdefault(
            task_name, {"metric": metric, "sample_score": []}
        )

        gt = row["output_field"]
        model_output = task_parsers[task_type].parse(outputs[row_idx])

        eval_fn = eval_methods[metric]
        metric_score = eval_fn(model_output, gt)
        per_task_metrics[task_name]["sample_score"].append(metric_score)
        per_task_metrics[task_name]["sample_score"].append(metric_score)

        if row_idx % print_interval == 0:
            print_sample(row_idx, outputs[row_idx], gt, metric, metric_score)

    # Aggregate scores
    for task_name in per_task_metrics:
        if per_task_metrics[task_name]["metric"] != "micro f1":
            per_task_metrics[task_name]["overall_metric"] = np.mean(
                per_task_metrics[task_name]["sample_score"]
            )
        else:
            per_task_metrics[task_name]["overall_metric"] = (
                metrics.compute_f1_score(
                    per_task_metrics[task_name]["sample_score"]
                )
            )

    print(per_task_metrics)

    overall_metrics = {"task_name": [], "metric": [], "overall_score": []}
    for task_name in per_task_metrics:
        overall_metrics["task_name"].append(task_name)
        overall_metrics["metric"].append(per_task_metrics[task_name]["metric"])
        overall_metrics["overall_score"].append(
            per_task_metrics[task_name]["overall_metric"]
        )

    overall_metrics = pd.DataFrame(overall_metrics)
    print(overall_metrics)
    overall_score = overall_metrics["overall_score"].mean()
    print(f"Overall Score: {overall_score}")
