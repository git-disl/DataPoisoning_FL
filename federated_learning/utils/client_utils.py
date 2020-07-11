def log_client_data_statistics(logger, label_class_set, distributed_dataset):
    """
    Logs all client data statistics.

    :param logger: logger
    :type logger: loguru.logger
    :param label_class_set: set of class labels
    :type label_class_set: list
    :param distributed_dataset: distributed dataset
    :type distributed_dataset: list(tuple)
    """
    for client_idx in range(len(distributed_dataset)):
        client_class_nums = {class_val : 0 for class_val in label_class_set}
        for target in distributed_dataset[client_idx][1]:
            client_class_nums[target] += 1

        logger.info("Client #{} has data distribution: {}".format(client_idx, str(list(client_class_nums.values()))))
