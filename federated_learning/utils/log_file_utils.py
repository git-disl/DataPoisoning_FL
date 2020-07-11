def get_poisoned_worker_ids_from_log(log_path):
    """
    :param log_path: string
    """
    with open(log_path, "r") as f:
        file_lines = [line.strip() for line in f.readlines()]

    workers = file_lines[3].split("[")[1].split("]")[0]
    workers = workers.replace(",", "")
    workers_list = workers.split(" ")

    return [int(worker) for worker in workers_list]
