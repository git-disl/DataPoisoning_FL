import random

def identify_random_elements(max, num_random_elements):
    """
    Picks a specified number of random elements from 0 - max.

    :param max: Max number to pick from
    :type max: int
    :param num_random_elements: Number of random elements to select
    :type num_random_elements: int
    :return: list
    """
    if num_random_elements > max:
        return []

    ids = []
    x = 0
    while x < num_random_elements:
        rand_int = random.randint(0, max - 1)

        if rand_int not in ids:
            ids.append(rand_int)
            x += 1

    return ids
