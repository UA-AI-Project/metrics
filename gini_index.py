from collections import Counter
import numpy as np


def calc_gini_index(x: np.array) -> float:
    """
    src: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797
    :param x: a numpy array of the values
    :return: gini index of array of values
    """
    if len(x) <= 1:
        return 0.0

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def get_gini_index(pred: dict[int, list[int]], k: int) -> float:
    """
    Compute the Gini index for all predictions by getting the number of item
    occurrences in the top k predictions.
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :return: gini index @ k
    """
    cntr = Counter()
    for preds in pred.values():
        top_k_preds = preds[:k]
        cntr += Counter(top_k_preds)

    # calculate the gini index based on the counter values
    vals = list(cntr.values())

    return calc_gini_index(np.asarray(vals))


if __name__ == '__main__':
    print(calc_gini_index(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 10000])))
    print(calc_gini_index(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))
