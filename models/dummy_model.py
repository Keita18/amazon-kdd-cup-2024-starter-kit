from typing import List
import random
import os

# please use this seed consistently across your code
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class DummyModel:
    """
    TODO
    """

    def __init__(self):
        """Initialize your models here"""
        random.seed(AICROWD_RUN_SEED)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        """
        Standard inferface for all tasks and tracks.

        The goal is for your model to be able to infer the task type,
        and respond with a string that is compatible with the task specific parser.


        Note: Even if the development dataset has the task_type information,
        During the actual evaluations, your code will only have access to the prompt,
        and the boolean variable indicating if its a multiple choice question.
        """

        potential_response = [1, 2, 3, 4]
        if is_multiple_choice:
            return str(random.choice(potential_response))
        else:
            # For Ranking, Retrieval, and Named Entity Recognition tasks
            # the expected response is a string that can be parsed with
            # `ast.literal_eval` (see parsers.py for more details)
            random.shuffle(potential_response)
            return str(potential_response)

            # Note: For the generation task, the expected response is a string
            # And, as this is a dummy response, we are just returning the
            # shuffled version of list, but in your case, it can be any string
