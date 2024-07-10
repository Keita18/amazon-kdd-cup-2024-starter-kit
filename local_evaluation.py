import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import metrics
import numpy as np
import pandas as pd
import parsers
import torch
from tqdm import tqdm
from time import time

VERSION = "0.1.0"


def print_sample(idx, generation, truth, metric, score):
    """
    Print a sample's generated output, the truth, and its evaluation score.
    """
    print(f"Sample {idx}, generation: {generation}")
    print(f"Sample {idx}, truth: {truth}")
    if isinstance(score, tuple) and len(score) == 3:
        print(
            f"Per Sample Metric Score ({metric}): tp {score[0]}, fp {score[1]}, fn {score[2]}"
        )
    else:
        print(f"Per Sample Metric Score ({metric}): {score}")
    print()


# Function to load development data from a JSON file
def load_development_data(filename):
    """
    Load development data from a specified JSON file.

    Parameters:
    - filename: Path to the JSON file containing the development data.

    Returns:
    - A pandas DataFrame containing the loaded data.
    """
    return pd.read_json(filename, lines=True)


# Function to generate model outputs based on the input data
def generate_model_outputs(data_df, model):
    """
    Generate predictions for each entry in the data DataFrame using a given model.

    Parameters:
    - data_df: A pandas DataFrame containing the input data for predictions.
    - model: The model instance used for generating predictions.

    Returns:
    - A list containing the model outputs for each entry in the data DataFrame.
    """
    outputs = []
    task_grouped_df = data_df.groupby(by=["task_type"])

    durations = []
    
    for task_type, task_group_data_df in task_grouped_df:
        task_group_data_df = task_group_data_df.reset_index(drop=True)
        
        is_multiple_choice = task_type[0] == "multiple-choice"
        batch_size = model.get_batch_size()
        
        batches = [task_group_data_df[i:i+batch_size] for i in range(0,len(task_group_data_df),batch_size)]
        
        for batch_df in batches:
            batch = {
                "prompt": batch_df["input_field"].tolist(),
            }
            start_time = time()
            model_output = model.batch_predict(
                    batch, 
                    is_multiple_choice
                )
            end_time = time()
            durations.append(end_time - start_time)
            outputs.append(
                pd.DataFrame({
                    "input_field": batch["prompt"],
                    "model_output_str": model_output
                }))

    average_time = sum(durations) / len(durations)
    print(f"Average time per batch: {average_time}")
    
    df_outputs = pd.concat(outputs)
    return df_outputs


# Function to evaluate the generated model outputs
def evaluate_outputs(data_df, log_every_n_steps=1):
    """
    Evaluate the model outputs against ground truth values using specified metrics.

    Parameters:
    - data_df: DataFrame containing the development data, including ground truth.
    - outputs: The generated outputs from the model to be evaluated.
    - log_every_n_steps: Logs samples every N steps

    Returns:
    - A dictionary containing evaluation metrics and scores for each task.
    """
    eval_methods = get_evaluation_methods()
    task_parsers = get_task_parsers()
    per_task_metrics = {}

    for row_idx, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Evaluating"
    ):
        task_name, task_type, metric, ground_truth, model_output_str = (
            row["task_name"],
            row["task_type"],
            row["metric"],
            row["output_field"],
            row["model_output_str"],
        )

        if metric not in eval_methods:
            raise NotImplementedError(f"No metric for {metric=}")

        model_output = task_parsers[task_type].parse(model_output_str)
        eval_fn = eval_methods[metric]
        metric_score = eval_fn(model_output, ground_truth)

        if task_name not in per_task_metrics:
            per_task_metrics[task_name] = {
                "task_type": task_type,
                "metric": metric,
                "sample_score": [],
            }

        per_task_metrics[task_name]["sample_score"].append(metric_score)

        if (row_idx + 1) % log_every_n_steps == 0:
            print_sample(
                row_idx + 1, model_output, ground_truth, metric, metric_score
            )

    return per_task_metrics


# Function to aggregate scores from evaluations
def aggregate_scores(per_task_metrics):
    """
    Aggregate evaluation scores across different tasks and metrics.

    Parameters:
    - per_task_metrics: A dictionary containing raw evaluation scores for each task.

    Returns:
    - A pandas DataFrame summarizing the overall metrics and scores.
    """
    overall_metrics = {
        "task_name": [],
        "task_type": [],
        "metric": [],
        "num_samples": [],
        "overall_score": [],
    }
    for task_name, values in per_task_metrics.items():
        task_type, metric, sample_scores = (
            values["task_type"],
            values["metric"],
            values["sample_score"],
        )
        overall_score = (
            np.mean(sample_scores)
            if metric != "micro f1"
            else metrics.calculate_f1_score(sample_scores)
        )

        overall_metrics["task_name"].append(task_name)
        overall_metrics["task_type"].append(task_type)
        overall_metrics["metric"].append(metric)
        overall_metrics["num_samples"].append(len(sample_scores))
        overall_metrics["overall_score"].append(overall_score)

    return pd.DataFrame(overall_metrics)


# Define and return evaluation methods
def get_evaluation_methods():
    """
    Get evaluation methods including accuracy, sentence transformers, and other metrics.

    Returns:
    - A dictionary mapping metric names to their respective evaluation functions.
    """
    return {
        "accuracy": metrics.calculate_per_sample_accuracy,
        "hit rate@3": metrics.calculate_hit_rate_3,
        "rougel": metrics.calculate_rougel,
        "sent-transformer": lambda generated_text, reference_texts: metrics.calculate_cosine_similarity(
            generated_text=generated_text,
            reference_texts=reference_texts,
            model_name="all-MiniLM-L6-v2",
        ),
        "multilingual-sent-transformer": lambda generated_text, reference_texts: metrics.calculate_cosine_similarity(
            generated_text=generated_text,
            reference_texts=reference_texts,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        ),
        "micro f1": metrics.calculate_true_positive_false_positives_false_negatives,
        "ndcg": metrics.calculate_ndcg,
        "bleu": metrics.calculate_bleu_score,
        "jp-bleu": lambda generated_text, reference_text: metrics.calculate_bleu_score(
            generated_text=generated_text,
            reference_text=reference_text,
            is_japanese=True,
        ),
    }


# Define and return task parsers
def get_task_parsers():
    """
    Define parsers for different task types to format model outputs accordingly.

    Returns:
    - A dictionary mapping task types to their respective parsers.
    """
    return {
        "multiple-choice": parsers.ShoppingBenchTaskParsers("multichoice"),
        "generation": parsers.ShoppingBenchTaskParsers("generation"),
        "retrieval": parsers.ShoppingBenchTaskParsers("retrieval"),
        "ranking": parsers.ShoppingBenchTaskParsers("ranking"),
        "named_entity_recognition": parsers.ShoppingBenchTaskParsers(
            "named_entity_recognition"
        ),
    }


# Main execution function to load data, generate model outputs, evaluate, and aggregate scores
def main():
    # Load development data
    # Please download the development data from : https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/dataset_files
    # and place it at: ./data/development.json
    DATA_FILENAME = "./data/corrected_development.json"

    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(
            f"Development data file not found at {DATA_FILENAME}."
            "Please download the development data from : https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/dataset_files"
            "and place it at: ./data/development.json"
        )

    data_df = load_development_data(DATA_FILENAME)

    # data_df = data_df[data_df["task_type"].isin(["multiple-choice"])]

    # Load the model from the user's custom configuration
    # Note: The evaluator **Always** imports the UserModel, please reference your own class
    # by setting the `UserModel` variable in models.user_config
    from models.user_config import UserModel

    model = UserModel()

    # Generate model outputs
    a = time()
    df_outputs = generate_model_outputs(data_df, model)
    b = time()
    
    # add outputs to the data_df
    merged_data_df = pd.merge(data_df, df_outputs, on="input_field")
        
    print(merged_data_df.head())

    # Evaluate the generated outputs and calculate metrics
    per_task_metrics = evaluate_outputs(merged_data_df)

    # Aggregate and display the evaluation scores
    overall_metrics = aggregate_scores(per_task_metrics)
    print("=" * 100)
    print("Task specific metrics: ")
    print(overall_metrics)

    print()
    # Calculate and print the overall score across all tasks and metrics
    overall_score = overall_metrics["overall_score"].mean()
    print(f"Overall Score: {overall_score}")

    overall_metrics["score"] = overall_metrics["num_samples"] * overall_metrics["overall_score"]
    overall_metrics["weight"] = overall_metrics["num_samples"]
    dt = overall_metrics.groupby("task_type")[["score", "weight"]].sum()
    dt["score"] = dt["score"] / dt["weight"]
    del dt["weight"]
    print(dt)

    print(f"Elapsed {b-a} seconds")


if __name__ == "__main__":
    main()
