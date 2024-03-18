class ShopBenchBaseModel:
    def __init__(self):
        pass

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
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        raise NotImplementedError("predict method not implemented")
