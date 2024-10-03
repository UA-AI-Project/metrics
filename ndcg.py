import math
import numpy as np


def get_ndcg_user(preds: list[int], truths: list[int], k: int,
                  idcg_ls: list[float], log2_inv: list[float]) -> float:
    """
    compute the NDCG for a list of predictions and truths
    :param preds: the prediction items of the user
    :param truths: the truth items of the user
    :param k: number of items to consider for the NDCG
    :return: NDCG@k for the user
    """
    # if there is no truth, return 1
    if not len(truths):
        return 1.0
    # if there are no predictions, return 0
    elif not len(preds):
        return 0.0

    if len(preds) < k:
        k = len(preds)

    # calculate idcg
    idcg = idcg_ls[min(len(truths), k) - 1]

    # calculate dcg
    truths_set = set(truths)
    dcg = 0
    for i, pred in enumerate(preds):
        if i == k:
            break

        if pred in truths_set:
            dcg += log2_inv[i]
            truths_set.remove(pred)

    return dcg / idcg


def get_ndcg(pred: dict[int, list[int]], truth: dict[int, list[int]],
             k: int) -> float:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) for a given prediction and truth.
    src: https://recpack.froomle.ai/generated/recpack.metrics.NDCGK.html
    :param pred: list of predicted items per user
    :param truth: list of ground truth items per user
    :param k: number of items to consider for the NDCG
    :return: NDCG@k
    """
    if k == 0:
        return 1.0

    # precompute a list of idcg
    log2_inv = [1 / math.log2(i + 2) for i in range(k)]
    idcg_ls = np.cumsum(log2_inv)

    # compute the NDCG per user
    total_ndcg = 0
    for user in pred.keys():
        if user not in truth:
            continue

        total_ndcg += get_ndcg_user(pred[user], truth[user], k, idcg_ls,
                                    log2_inv)

    return total_ndcg / len(truth.keys())
