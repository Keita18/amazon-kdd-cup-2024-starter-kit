import ast

from loguru import logger

VERSION = "0.1.1"

MAX_RESPONSE_CHARACTERS = 5000


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

        assert isinstance(
            response, str
        ), f"Response must be a string, but got {type(response)}"

        # Consider only the first MAX_RESPONSE_CHARACTERS
        response = response[:MAX_RESPONSE_CHARACTERS]

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
        default_response = -1
        try:
            response = response.strip()
            return int(response[0])
        except Exception as e:
            logger.warning(
                f"SHOPBENCH_PARSER_WARNING::: Error parsing multichoice response: {e}. Responding with default : {default_response}"
            )
            return default_response

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
        default_respomse = []
        # Keep only numeric characters and specific punctuation.
        cleaned_response = "".join(
            c for c in response if c.isnumeric() or c in [",", " "]
        )

        # Convert to list of integers
        ranked_items = []
        for item in cleaned_response.split(","):
            try:
                # Attempt to convert each item to an integer and add it to the list.
                int_item = int(item)
                if int_item <= 5:  # we know int_item can be at most 5
                    ranked_items.append(int_item)
            except ValueError:
                pass  # Skip non-numeric items.

        # Consider only the first 5 unique elements.
        ranked_items = ranked_items[:5]

        # If there are duplicates, empty the list
        if len(ranked_items) != len(set(ranked_items)):
            ranked_items = default_respomse
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
        default_response = []
        try:
            # Similar to ranking parser, but only returns the first 3 elements.
            cleaned_response = "".join(
                c for c in response if c.isnumeric() or c in [",", " "]
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
        except Exception as e:
            logger.warning(
                f"SHOPBENCH_PARSER_WARNING::: Error parsing retrieval response: {e}. Responding with default : {default_response}"
            )
            return default_response

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
            else:
                raise SyntaxError(
                    "Unexpected Syntax error - fall back to comma separated list."
                )
        except Exception as e:
            # Fallback: split the string by commas and strip whitespace.
            # we remove empty entities. it will not cause bug, just an implementation choice.
            return [
                entity.strip()
                for entity in response.split(",")
                if entity.strip() != ""
            ]


import unittest


class TestShoppingBenchTaskParsers(unittest.TestCase):

    def test_multichoice(self):
        parser = ShoppingBenchTaskParsers("multichoice")
        # Check for a valid numeric response
        self.assertEqual(parser.parse("2"), 2)
        # Check for an invalid (alphabetic) response, expecting failure code -1
        self.assertEqual(parser.parse("a"), -1)
        # Check handling of newline-only input, expecting failure code -1
        self.assertEqual(parser.parse("\n"), -1)
        # Check handling of space-only input, expecting failure code -1
        self.assertEqual(parser.parse(" "), -1)
        # Check handling of leading space before a valid response
        self.assertEqual(parser.parse(" 2"), 2)
        # Check handling of newline before a valid response
        self.assertEqual(parser.parse("\n1"), 1)
        # Check for newline and space before a valid response
        self.assertEqual(parser.parse("\n 3"), 3)
        # Check for newline and space only, expecting failure code -1
        self.assertEqual(parser.parse("\n "), -1)

    def test_ranking(self):
        parser = ShoppingBenchTaskParsers("ranking")
        # Basic successful parse of a comma-separated list of numbers
        self.assertEqual(parser.parse("1, 2, 3, 4, 5"), [1, 2, 3, 4, 5])
        # Successfully parses even when wrapped in square brackets
        self.assertEqual(parser.parse("[1, 2, 3, 4, 5]"), [1, 2, 3, 4, 5])
        # Fails (empty list) when numbers are repeated
        self.assertEqual(parser.parse("1, 2, 2, 3"), [])
        # Filters out non-numeric values correctly, keeping the valid numbers
        self.assertEqual(parser.parse("1, 2, 4, aicrowd, 5"), [1, 2, 4, 5])
        # Check handling of newline-only input, expecting empty list
        self.assertEqual(parser.parse("\n"), [])
        # Check handling of space and newline input, expecting empty list
        self.assertEqual(parser.parse(" \n"), [])
        # Parses numbers correctly even when prefixed by non-numeric text
        self.assertEqual(
            parser.parse("The answer is: 1, 2, 3, 4, 5"), [1, 2, 3, 4, 5]
        )
        # Correctly handles a leading comma
        self.assertEqual(parser.parse(",1,2,3,4,5"), [1, 2, 3, 4, 5])
        # Fails (empty list) when numbers are not comma-separated
        self.assertEqual(parser.parse("1 2"), [])

    def test_generation(self):
        parser = ShoppingBenchTaskParsers("generation")
        # Verifies correct response without modification
        self.assertEqual(
            parser.parse("This is a generated response."),
            "This is a generated response.",
        )
        # Handles and trims extraneous newlines and spaces correctly
        self.assertEqual(
            parser.parse("\nThe answer is \n\n good.\n\n\n\n\n\n\n"),
            "The answer is \n\n good.",
        )
        # Correctly returns empty string for newline and space-only inputs
        self.assertEqual(parser.parse("\n \n"), "")

    def test_retrieval(self):
        parser = ShoppingBenchTaskParsers("retrieval")
        # Basic successful parse of a comma-separated list of numbers
        self.assertEqual(parser.parse("100, 200, 300"), [100, 200, 300])
        # Successfully handles shorter than expected input lists
        self.assertEqual(parser.parse("100, 200"), [100, 200])
        # Filters out non-numeric values correctly, keeping the valid numbers
        self.assertEqual(parser.parse("100, 200, jjhg"), [100, 200])
        # Correctly parses numbers despite excessive spacing and newlines
        self.assertEqual(
            parser.parse("100,           200, \n\n\n 300"), [100, 200, 300]
        )
        # Limits output to first three elements if more are provided
        self.assertEqual(parser.parse("100, 200, 300, 400"), [100, 200, 300])
        # Correctly handles newline before valid input
        self.assertEqual(parser.parse("\n 100, 200, 300"), [100, 200, 300])
        # Returns empty list for newline-only inputs
        self.assertEqual(parser.parse("\n \n \n"), [])

    def test_named_entity_recognition(self):
        parser = ShoppingBenchTaskParsers("named_entity_recognition")
        # Successfully parses a list of strings, correctly interpreting them as separate entities
        self.assertEqual(
            parser.parse("['New York', 'ShopBench', 'Amazon']"),
            ["New York", "ShopBench", "Amazon"],
        )
        # Successfully parses comma-separated entities without brackets or quotes
        self.assertEqual(
            parser.parse("New York, ShopBench, Amazon"),
            ["New York", "ShopBench", "Amazon"],
        )
        # Incorrectly includes the opening bracket in the first entity and the closing bracket in the last entity,
        # indicating an unintentional parsing error with brackets when quotes are not used.
        self.assertEqual(
            parser.parse("[New York, ShopBench, Amazon]"),
            ["[New York", "ShopBench", "Amazon]"],
        )
        # Correctly parses entities even when the input starts with a newline and a comma, trimming unnecessary characters
        self.assertEqual(
            parser.parse("\n, New York, ShopBench"), ["New York", "ShopBench"]
        )
        # Returns an empty list when parsing only a space, indicating no entities found
        self.assertEqual(parser.parse(" "), [])
        # Returns an empty list for inputs consisting only of newlines and spaces, indicating no entities found
        self.assertEqual(parser.parse("\n \n"), [])


if __name__ == "__main__":
    unittest.main()
