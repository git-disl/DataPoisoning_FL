from .selection_strategy import SelectionStrategy
import random

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])
