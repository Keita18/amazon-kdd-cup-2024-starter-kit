from typing import List


class DummyModel:
    """
    Note to participants:
        Example class to show the different functions to be implemented for each type of task
        Make sure to follow the data types as mentioned in the function definitions
    """
    def __init__(self):
        """ Initialize your models here """
        pass
    
    def task_multichoice(self, task_prompt: str) -> int:
        """
        Task method for Multiple choice questions
            Input - Task Prompt (includes choices)
            Output - Single integer index among ones given in the input
        """
        return 0

    def task_ranking(self, task_prompt: str) -> List[float]:
        """
        Task method for Ranking
            Input - Task Prompt (includes items to rank)
            Output - Ordered List of ranks for each item
        """
        return [1, 0, 2, 3]

    def task_generation(self, task_prompt: str) -> str:
        """
        Task method for Generation
            Input - Task Prompt describing the required generation
            Output - Generated text as per task prompt
        """
        return "This is a test"

    def task_retrieval(self, task_prompt: str) -> List[int]:
        """
       Task method for Generation
            Input - Task Prompt describing the items which need to be selected from (includes indexes of items)
            Output - Unordered list of indexes selected (must be a python list even if single item)
        """
        return [0, 1, 2]

    def task_named_entity_recognition(self, task_prompt: str) -> List[str]:
        """
        Task method for Named Entity Recognition
            Input - Task Prompt describing the named entity recognition task
            Output - Unordered list of one or more entity names (must be a python list even if single item)
        """
        return ["food", "gpu"]