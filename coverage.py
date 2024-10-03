def get_coverage(pred: dict[int, list[int]], n_items: int, k: int) -> float:
    """
    Compute the coverage of a given prediction.
    src: https://dl.acm.org/doi/pdf/10.1145/1060745.1060754 (3.2.1)
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param n_items: number of all possible items
    :param k: number of items to consider in the prediction
    :return: coverage@k
    """
    items = set()
    for user in pred.keys():
        for i in range(min(k, len(pred[user]))):
            items.add(pred[user][i])

    return len(items) / n_items
