import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import os

from sentence_transformers import SentenceTransformer
import metrics
import parsers


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
    for _, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Generating Responses"
    ):
        is_multiple_choice = row["task_type"] == "multiple-choice"
        prompt = row["input_field"]
        model_output = model.predict(prompt, is_multiple_choice)
        outputs.append(model_output)
    return outputs


# Function to evaluate the generated model outputs
def evaluate_outputs(data_df, outputs, log_every_n_steps=1):
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
        task_type, metric, ground_truth = (
            row["task_type"],
            row["metric"],
            row["output_field"],
        )

        if metric not in eval_methods:
            raise NotImplementedError(f"No metric for {metric=}")

        task_name = f"{task_type}---{metric}"
        # Note: In practice, here we are using the task_type-metric pair as a unique identifier, calling it as the task_name.
        # During the actual evaluations, the task names are more semantically defined, meaning, there could be multiple tasks
        # with the same task_type and metric.

        model_output = task_parsers[task_type].parse(outputs[row_idx])
        eval_fn = eval_methods[metric]
        metric_score = eval_fn(model_output, ground_truth)

        if task_name not in per_task_metrics:
            per_task_metrics[task_name] = {
                "task_type": task_type,
                "metric": metric,
                "sample_score": [],
            }

        per_task_metrics[task_name]["sample_score"].append(metric_score)

        if row_idx % log_every_n_steps == 0:
            print_sample(
                row_idx, model_output, ground_truth, metric, metric_score
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
            else metrics.compute_f1_score(sample_scores)
        )

        overall_metrics["task_name"].append(task_name)
        overall_metrics["task_type"].append(task_type)
        overall_metrics["metric"].append(metric)
        overall_metrics["overall_score"].append(overall_score)

    return pd.DataFrame(overall_metrics)


# Define and return evaluation methods
def get_evaluation_methods():
    """
    Get evaluation methods including accuracy, sentence transformers, and other metrics.

    Returns:
    - A dictionary mapping metric names to their respective evaluation functions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_all_lm = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    sentence_multilingual = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ).to(device)

    return {
        "accuracy": metrics.accuracy,
        "hit rate@3": metrics.hit_rate_3,
        "rougel": metrics.rougel,
        "sent-transformer": lambda g, t: metrics.sent_transformer(
            g, t, sentence_all_lm
        ),
        "multilingual-sent-transformer": lambda g, t: metrics.sent_transformer(
            g, t, sentence_multilingual
        ),
        "micro f1": metrics.tp_fp_fn,
        "ndcg": metrics.ndcg_eval,
        "bleu": metrics.bleu,
        "jp-bleu": lambda g, t: metrics.bleu(g, t, jp=True),
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
    # Please download the development data from : https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/dataset_files
    # and place it at: ./data/development.json
    DATA_FILENAME = "./data/development.json"

    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(
            f"Development data file not found at {DATA_FILENAME}."
            "Please download the development data from : https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/dataset_files"
            "and place it at: ./data/development.json"
        )

    data_df = load_development_data(DATA_FILENAME)

    # Load the model from the user's custom configuration
    # Note: The evaluator **Always** imports the UserModel, please reference your own class
    # by setting the `UserModel` variable in models.user_config
    from models.user_config import UserModel

    model = UserModel()

    # Generate model outputs
    outputs = generate_model_outputs(data_df, model)
    data_df["outputs"] = (
        outputs  # Optional: Add outputs back to DataFrame for inspection
    )
    print(data_df.head())

    # Evaluate the generated outputs and calculate metrics
    per_task_metrics = evaluate_outputs(data_df, outputs)

    # Aggregate and display the evaluation scores
    overall_metrics = aggregate_scores(per_task_metrics)
    print("=" * 100)
    print("Task specific metrics: ")
    print(overall_metrics)

    print()
    # Calculate and print the overall score across all tasks and metrics
    overall_score = overall_metrics["overall_score"].mean()
    print(f"Overall Score: {overall_score}")


if __name__ == "__main__":
    main()
