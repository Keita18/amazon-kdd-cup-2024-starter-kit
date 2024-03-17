import ast


class ShoppingBenchTaskParsers:
    """
    A class designed to parse responses from different task types in
    the ShopBench - MultiTask Online Shopping Challenge for LLMs.
    It supports a variety of task types such as multiple choice, ranking, generation, retrieval,
    and named entity recognition, each with its own specific parsing logic to format the raw
    response strings into structured data.

    Attributes:
        task_type (str): The type of task the parser is set up to handle. Valid task types
                         include 'multichoice', 'ranking', 'generation', 'retrieval',
                         and 'named_entity_recognition'.
    """

    def __init__(self, task_type: str) -> None:
        """
        Initializes the parser for a specific task type.

        Parameters:
            task_type (str): Specifies the task type this parser instance will handle.
        """
        self.task_type = task_type

    def parse(self, response: str) -> any:
        """
        Parses a given response string according to the task type of the parser, and returns
        a structured representation of that response.

        Parameters:
            response (str): The raw response string obtained from performing the task.

        Returns:
            A parsed and appropriately formatted response suitable for the parser's task type.
            The format of the return value varies with the task type.
        """
        # Map of task types to their corresponding parsing methods.
        task_parser_methods = {
            "multichoice": self._parse_multichoice,
            "ranking": self._parse_ranking,
            "generation": self._parse_generation,
            "retrieval": self._parse_retrieval,
            "named_entity_recognition": self._parse_named_entity_recognition,
        }

        # Attempt to retrieve the appropriate parser method for the task type.
        parser_method = task_parser_methods.get(self.task_type)

        # Execute the parser method if found, otherwise raise an error.
        if parser_method:
            return parser_method(response)
        else:
            raise NotImplementedError(
                f"Task type '{self.task_type}' is not supported."
            )

    def _parse_multichoice(self, response: str) -> int:
        """
        Parses a response from a multiple-choice task.

        Assumes the first character of the response string indicates the chosen option.

        Parameters:
            response (str): The raw response string.

        Returns:
            An integer representing the selected option. Returns -1 if the parsing fails due to
            an invalid response format.
        """
        try:
            return int(response.strip()[0])
        except ValueError:
            return -1

    def _parse_ranking(self, response: str) -> list:
        """
        Parses a ranking task response into a list of ranked items.

        Expects a string with numeric values separated by commas, indicating the ranking order.

        Parameters:
            response (str): The raw response string.

        Returns:
            A list of integers representing the items in ranked order. Limits to the first 5 unique
            elements. Returns an empty list if duplicates are found or parsing fails.
        """
        # Keep only numeric characters and specific punctuation.
        cleaned_response = "".join(
            c for c in response if c.isnumeric() or c in ["[", "]", ",", " "]
        )

        # Convert to list of integers
        ranked_items = []
        for item in cleaned_response.split(","):
            try:
                # Attempt to convert each item to an integer and add it to the list.
                ranked_items.append(int(item))
            except ValueError:
                pass  # Skip non-numeric items.

        # Consider only the first 5 unique elements.
        ranked_items = ranked_items[:5]

        # If there are duplicates, empty the list
        if len(ranked_items) != len(set(ranked_items)):
            ranked_items = []
        return ranked_items

    def _parse_generation(self, response: str) -> str:
        """
        Parses a response from a generation task by trimming whitespace.

        This method primarily cleans up the response string for presentation or further processing.

        Parameters:
            response (str): The raw response string.

        Returns:
            A trimmed version of the response string.
        """
        return response.strip()

    def _parse_retrieval(self, response: str) -> list:
        """
        Parses a retrieval task response, extracting the identifiers of retrieved items.

        The response is expected to contain numeric values separated by commas.

        Parameters:
            response (str): The raw response string.

        Returns:
            A list of integers representing the first 3 unique retrieved item indices.
        """
        # Similar to ranking parser, but only returns the first 3 elements.
        cleaned_response = "".join(
            c for c in response if c.isnumeric() or c in ["[", "]", ",", " "]
        )

        # Convert to list of integers
        response = []
        for item in cleaned_response.split(","):
            try:
                # Attempt to convert each item to an integer and add it to the list.
                response.append(int(item))
            except ValueError:
                pass  # Skip non-numeric items.

        # consider only the first 3 elements
        retrieved_items = response[:3]

        return retrieved_items

    def _parse_named_entity_recognition(self, response: str) -> list:
        """
        Parses a response from a named entity recognition (NER) task.

        Can handle both list-like string inputs or comma-separated entities in a plain string.

        Parameters:
            response (str): The raw response string.

        Returns:
            A list of named entities extracted from the response. Attempts to parse the response as a
            literal list; falls back to splitting by commas if that fails.
        """
        try:
            # Attempt to interpret the response as a literal list.
            entities = ast.literal_eval(response)
            if isinstance(entities, list) and all(
                isinstance(item, str) for item in entities
            ):
                return entities
        except (SyntaxError, ValueError):
            # Fallback: split the string by commas and strip whitespace.
            return [entity.strip() for entity in response.split(",")]


if __name__ == "__main__":
    # Example usage of the ShoppingBenchTaskParsers class for various task types.

    # MULTICHOICE EXAMPLE
    multic_choice_parser = ShoppingBenchTaskParsers("multichoice")
    print("Multichoice Example:")
    print(multic_choice_parser.parse("2"))  # Expected output: 2
    print(
        multic_choice_parser.parse("a")
    )  # Expected output (failure case): -1
    print()

    # RANKING EXAMPLE
    ranking_parser = ShoppingBenchTaskParsers("ranking")
    print("Ranking Example:")
    print(
        ranking_parser.parse("1, 2, 3, 4, 5")
    )  # Expected output: [1, 2, 3, 4, 5]
    print(
        ranking_parser.parse("[1, 2, 2, 3]")
    )  # Expected output (failure case): [] # because of repeating numbers
    print(
        ranking_parser.parse("1, 4, 5, aicrowd, 6")
    )  # Expected output: [1, 4, 5, 6] # remove alphanumeric chars

    print()

    # GENERATION EXAMPLE
    generation_parser = ShoppingBenchTaskParsers("generation")
    print("Generation Example:")
    print(
        generation_parser.parse("This is a generated response")
    )  # Expected output: 'This is a generated response.'
    print()

    # RETRIEVAL EXAMPLE
    retrieval_parser = ShoppingBenchTaskParsers("retrieval")
    print("Retrieval Example:")
    print(
        retrieval_parser.parse("100, 200, 300")
    )  # Expected output: [100, 200, 300]
    print(
        retrieval_parser.parse("100, 200")
    )  # Expected output (shorter than 3): [100, 200]
    print(
        retrieval_parser.parse("100, 200, jjhg")
    )  # Expected output (removed alphhanumeric chars): [100, 200]
    print(
        retrieval_parser.parse("100, 200, 300, 400")
    )  # Expected output (only consider first 3 elems): [100, 200, 300]

    print()

    # NAMED ENTITY RECOGNITION EXAMPLE
    ner_parser = ShoppingBenchTaskParsers("named_entity_recognition")
    print("Named Entity Recognition Example:")
    print(
        ner_parser.parse("['New York', 'ShopBench', 'Amazon']")
    )  # Expected output: ['New York', 'ShopBench', 'Amazon']
    print(
        ner_parser.parse("New York, ShopBench, Amazon")
    )  # Expected output: ['New York', 'ShopBench', 'Amazon']
    print(
        ner_parser.parse("[New York, ShopBench, Amazon]")
    )  # Expected output (failure case - extra '[' characters added to boundary elems]): ['[New York', 'ShopBench', 'Amazon]']
