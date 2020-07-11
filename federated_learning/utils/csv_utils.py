def convert_results_to_csv(results):
    """
    :param results: list(return data by test_classification() in client.py)
    """
    cleaned_epoch_test_set_results = []

    for row in results:
        components = [row[0], row[1]]

        for class_precision in row[2]:
            components.append(class_precision)
        for class_recall in row[3]:
            components.append(class_recall)

        cleaned_epoch_test_set_results.append(components)

    return cleaned_epoch_test_set_results
