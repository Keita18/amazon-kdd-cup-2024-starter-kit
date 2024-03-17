#!/usr/bin/env python3
import ast


class ShoppingBenchTaskParsers:
    """
    A class for parsing responses from different types of tasks in a shopping bench scenario.

    Attributes:
        task_type (str): The type of task for which the parser is instantiated.
    """

    def __init__(self, task_type: str) -> None:
        """
        Initializes the parser with a specific task type.

        Parameters:
            task_type (str): The type of task, e.g., 'multichoice', 'ranking', etc.
        """
        self.task_type = task_type

    def parse(self, response: str) -> any:
        """
        Parses the response based on the task type.

        Parameters:
            response (str): The raw response string from the model.

        Returns:
            The parsed response, formatted according to the task type's requirements.
        """
        # Mapping task types to their respective parsing methods.
        task_parser_methods = {
            "multichoice": self._task_multichoice_parser,
            "ranking": self._task_ranking_parser,
            "generation": self._task_generation_parser,
            "retrieval": self._task_retrieval_parser,
            "named_entity_recognition": self._task_named_entity_recognition_parser,
        }

        # Retrieve the parser method based on the task type.
        parser_method = task_parser_methods.get(self.task_type)

        if parser_method is not None:
            return parser_method(response)
        else:
            raise NotImplementedError(
                f"Task type {self.task_type} not implemented"
            )

    def _task_multichoice_parser(self, response: str) -> int:
        """
        Parses a multichoice task response.

        Parameters:
            response (str): A string representing the selected option's index.

        Returns:
            int: The index of the selected option, or -1 if the input is invalid.
        """
        try:
            return int(response.strip())
        except ValueError:
            return -1

    def _task_ranking_parser(self, response: str) -> list:
        """
        Parses a ranking task response.

        Parameters:
            response (str): A string representing the ordered list of ranks.

        Returns:
            list: A list of ranks if the input is valid, otherwise ignore non numeric list elements.
        """
        return self._parse_list(response, expected_type=float)

    def _task_generation_parser(self, response: str) -> str:
        """
        Parses a generation task response.

        Parameters:
            response (str): The generated text response.

        Returns:
            str: The stripped response text.
        """
        return response.strip()

    def _task_retrieval_parser(self, response: str) -> list:
        """
        Parses a retrieval task response.

        Parameters:
            response (str): A string representing the indexes of selected items.

        Returns:
            list: A list of selected item indexes if the input is valid, otherwise ignore non numeric list elements.
        """
        return self._parse_list(response, expected_type=int)

    def _task_named_entity_recognition_parser(self, response: str) -> list:
        """
        Parses a named entity recognition task response.

        Parameters:
            response (str): A string representing the list of identified entities.

        Returns:
            list: A list of entity names if the input is valid.
        """
        return self._parse_list(response, expected_type=str)

    def _parse_list(self, response: str, expected_type: type) -> list:
        """
        A helper method to parse a string into a list with elements of an expected type.

        Parameters:
            response (str): The string to parse.
            expected_type (type): The expected type of elements in the list.

        Returns:
            list: A list of elements of the expected type, or ignore items if parsing fails.
        """
        try:
            parsed_response = ast.literal_eval(response)
            if not isinstance(parsed_response, list):
                return []

            sanitized_response = []
            for item in parsed_response:
                try:
                    sanitized_response.append(expected_type(item))
                except (ValueError, TypeError) as e:
                    pass
            return sanitized_response
        except SyntaxError:
            return []


if __name__ == "__main__":
    # This section demonstrates the use of the ShoppingBenchTaskParsers class
    # for different types of tasks. For each task, we initialize a parser,
    # provide it with a response string, and then output the parsed result.

    # MULTICHOICE TASK EXAMPLE
    # Initialize the parser for a multichoice task
    multichoice_parser = ShoppingBenchTaskParsers("multichoice")
    # Example response string for a multichoice task (correct option is 2)
    multichoice_response = "2"
    # Parse the response and print the result
    print(
        "Multichoice Task Parsing Result:",
        multichoice_parser.parse(multichoice_response),
    )
    # Expected output: 2

    # RANKING TASK EXAMPLE
    # Initialize the parser for a ranking task
    ranking_parser = ShoppingBenchTaskParsers("ranking")
    # Example response string for a ranking task (items ranked as 3rd, 1st, 2nd)
    ranking_response = "[3, 1, 2]"
    # Parse the response and print the result
    print(
        "Ranking Task Parsing Result:", ranking_parser.parse(ranking_response)
    )
    # Expected output: [3.0, 1.0, 2.0]

    # GENERATION TASK EXAMPLE
    # Initialize the parser for a text generation task
    generation_parser = ShoppingBenchTaskParsers("generation")
    # Example response string for a generation task
    generation_response = (
        "This is a generated response based on the input prompt."
    )
    # Parse the response and print the result
    print(
        "Generation Task Parsing Result:",
        generation_parser.parse(generation_response),
    )
    # Expected output: This is a generated response based on the input prompt.

    # RETRIEVAL TASK EXAMPLE
    # Initialize the parser for a retrieval task
    retrieval_parser = ShoppingBenchTaskParsers("retrieval")
    # Example response string for a retrieval task (items at indexes 0 and 2 are relevant)
    retrieval_response = "[0, 2]"
    # Parse the response and print the result
    print(
        "Retrieval Task Parsing Result:",
        retrieval_parser.parse(retrieval_response),
    )
    # Expected output: [0, 2]

    # NAMED ENTITY RECOGNITION (NER) TASK EXAMPLE
    # Initialize the parser for a named entity recognition task
    ner_parser = ShoppingBenchTaskParsers("named_entity_recognition")
    # Example response string for an NER task
    ner_response = '["New York", "ShopBench"]'
    # Parse the response and print the result
    print("NER Task Parsing Result:", ner_parser.parse(ner_response))
    # Expected output: ['New York', 'ShopBench']

    # This demonstrates the flexible and effective parsing capabilities of the
    # ShoppingBenchTaskParsers class across a variety of task types.

    # Failure Case Examples for ShoppingBenchTaskParsers
    # These examples illustrate how the parser handles incorrect or unexpected inputs.

    print("=== FAILURE CASES ===\n")

    # MULTICHOICE TASK FAILURE EXAMPLE
    # Non-integer response for a multichoice task
    multichoice_parser = ShoppingBenchTaskParsers("multichoice")
    multichoice_bad_response = "abc"  # Invalid response (not an integer)
    print(
        "Multichoice Task Failure Case:",
        multichoice_parser.parse(multichoice_bad_response),
    )
    # Expected output: -1 (indicating an invalid response)

    # RANKING TASK FAILURE EXAMPLE
    # Non-list response for a ranking task
    ranking_parser = ShoppingBenchTaskParsers("ranking")
    ranking_bad_response = "not a valid list"  # Invalid list format
    print(
        "Ranking Task Failure Case:",
        ranking_parser.parse(ranking_bad_response),
    )
    # Expected output: [] (indicating an inability to parse the response)

    # GENERATION TASK FAILURE EXAMPLE
    # Empty or whitespace-only response for a generation task
    generation_parser = ShoppingBenchTaskParsers("generation")
    generation_bad_response = "    "  # Only spaces
    print(
        "Generation Task Failure Case:",
        f"'{generation_parser.parse(generation_bad_response)}'",
    )
    # Expected output: '' (an empty string indicating an invalid or empty response)

    # RETRIEVAL TASK FAILURE EXAMPLE
    # Incorrect element format for a retrieval task
    retrieval_parser = ShoppingBenchTaskParsers("retrieval")
    retrieval_bad_response = "[1, 'a']"  # Contains a non-integer
    print(
        "Retrieval Task Failure Case:",
        retrieval_parser.parse(retrieval_bad_response),
    )
    # Expected output: [1] (ignores invalid non-integer values)

    # NAMED ENTITY RECOGNITION (NER) TASK FAILURE EXAMPLE
    # Non-list or incorrect entity format for an NER task
    ner_parser = ShoppingBenchTaskParsers("named_entity_recognition")
    ner_bad_response = '{"entity": "New York"}'  # Not a list, incorrect format
    print("NER Task Failure Case:", ner_parser.parse(ner_bad_response))
    # Expected output: [] (indicating the response could not be parsed as a list of entities)

    print(
        "\nThese examples demonstrate how the parser handles various incorrect inputs."
    )
