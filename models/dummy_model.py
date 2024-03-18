from typing import List, Union
import random
import os

from .base_model import ShopBenchBaseModel

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class DummyModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        """
        Generates a prediction based on the input prompt and task type.

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Args:
            prompt (str): The input prompt for the model.
            is_multiple_choice (bool): Indicates whether the task is a multiple choice question.

        Returns:
            str: The prediction as a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        possible_responses = [1, 2, 3, 4]

        if is_multiple_choice:
            # Randomly select one of the possible responses for multiple choice tasks
            return str(random.choice(possible_responses))
        else:
            # For other tasks, shuffle the possible responses and return as a string
            random.shuffle(possible_responses)
            return str(possible_responses)
            # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
            # For generation tasks, this should ideally return an unconstrained string.
