from .selection_strategy import SelectionStrategy
import random

class AfterBreakpoint(SelectionStrategy):
    """
    Will not select poisoned workers at and after the break point epoch, but will select the
    poisoned workers before the break point epoch.
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        breakpoint_epoch = kwargs["AfterBreakPoint_EPOCH"]
        num_workers = kwargs["AfterBreakpoint_NUM_WORKERS_PER_ROUND"]
        current_epoch_number = kwargs["current_epoch_number"]

        selected_workers = []
        if current_epoch_number < breakpoint_epoch:
            selected_workers = random.sample(workers, num_workers)
        else:
            non_poisoned_workers = list(set(workers) - set(poisoned_workers))

            selected_workers = random.sample(non_poisoned_workers, num_workers)

        return selected_workers
