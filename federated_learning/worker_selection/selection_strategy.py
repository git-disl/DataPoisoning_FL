from abc import abstractmethod

class SelectionStrategy:

    @abstractmethod
    def select_round_workers(self, workers, poisoned_workers, kwargs):
        """
        :param workers: list(int). All workers available for learning
        :param poisoned_workers: list(int). All workers that are poisoned
        :param kwargs: dict
        """
        raise NotImplementedError("select_round_workers() not implemented")
