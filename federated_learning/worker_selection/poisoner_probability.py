from .selection_strategy import SelectionStrategy
import random
import copy

class PoisonerProbability(SelectionStrategy):
    """
    Will not select poisoned workers before or after a specified epoch (specified in arguments).

    Will artificially boost / reduce likelihood of the poisoned workers being selected.
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        break_epoch = kwargs["PoisonerProbability_BREAK_EPOCH"]
        post_break_epoch_probability = kwargs["PoisonerProbability_POST_BREAK_EPOCH_PROBABILITY"]
        pre_break_epoch_probability = kwargs["PoisonerProbability_PRE_BREAK_EPOCH_PROBABILITY"]
        num_workers = kwargs["PoisonerProbability_NUM_WORKERS_PER_ROUND"]
        current_epoch_number = kwargs["current_epoch_number"]

        workers = self.remove_poisoned_workers_from_group(poisoned_workers, workers)

        selected_workers = []
        if current_epoch_number >= break_epoch:
            selected_workers = self.select_workers(num_workers, post_break_epoch_probability, poisoned_workers, workers)
        else:
            selected_workers = self.select_workers(num_workers, pre_break_epoch_probability, poisoned_workers, workers)

        return selected_workers

    def remove_poisoned_workers_from_group(self, poisoned_workers, group):
        """
        Removes all instances of set(poisoned_workers) from set(group).
        """
        return list(set(group) - set(poisoned_workers))

    def select_workers(self, num_workers, probability_threshold, group_0, group_1):
        """
        Selects a set of workers from the two different groups.

        Weights the choice via the probability threshold
        """
        group_0_copy = copy.deepcopy(group_0)
        group_1_copy = copy.deepcopy(group_1)

        selected_workers = []
        while len(selected_workers) < num_workers:
            group_to_select_worker_from = self.select_group(probability_threshold, group_0, group_1)
            selected_worker = random.choice(group_to_select_worker_from)
            if selected_worker not in selected_workers:
                selected_workers.append(selected_worker)

        return selected_workers

    def select_group(self, probability_threshold, group_0, group_1):
        """
        Selects between group_0 and group_1 based on a random choice.

        Probability threshold determines weighting given to group 0.
        Ex: if 0 is the probability threshold, then group 0 will never be selected.
        """
        next_int = random.uniform(0, 1)

        if next_int <= probability_threshold:
            return group_0
        else:
            return group_1

if __name__ == '__main__':
    selector = PoisonerProbability()

    print(selector.select_round_workers([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], [3,4,5,6,10,11,12], {
        "PoisonerProbability_BREAK_EPOCH" : 5,
        "PoisonerProbability_POST_BREAK_EPOCH_PROBABILITY" : 0.0,
        "PoisonerProbability_PRE_BREAK_EPOCH_PROBABILITY" : 1.0,
        "PoisonerProbability_NUM_WORKERS_PER_ROUND" : 5,
        "current_epoch_number" : 10
    }))
