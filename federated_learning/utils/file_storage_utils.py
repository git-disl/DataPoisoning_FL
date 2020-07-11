import csv
import json

def generate_json_repr_for_worker(worker_id, is_worker_poisoned, test_set_results):
    """
    :param worker_id: int
    :param is_worker_poisoned: boolean
    :param test_set_results: list(dict)
    """
    return {
        "worker_id" : worker_id,
        "is_worker_poisoned" : is_worker_poisoned,
        "test_set_results" : test_set_results
    }

def convert_test_results_to_json(epoch_idx, accuracy, loss, class_precision, class_recall):
    """
    :param epoch_idx: int
    :param accuracy: float
    :param loss: float
    :param class_precision: list(float)
    :param class_recall: list(float)
    """
    return {
        "epoch" : epoch_idx,
        "accuracy" : accuracy,
        "loss" : loss,
        "class_precision" : class_precision,
        "class_recall" : class_recall
    }

def save_results(results, filename):
    """
    :param results: experiment results
    :type results: list()
    :param filename: File name to write results to
    :type filename: String
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for experiment in results:
            writer.writerow(experiment)

def read_results(filename):
    """
    :param filename: File name to read results from
    :type filename: String
    """
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            data.append(row)

    return data

def save_results_v2(results, filename):
    """
    Save results to a file. Using format v2.

    :param results: json
    :param filename: string
    """
    with open(filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

def read_results_v2(filename):
    """
    Read results from a file. Using format v2.
    """
    with open(filename, "r") as f:
        return json.load(f)
