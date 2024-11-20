import sklearn.metrics
import numpy as np


def _get_intra_list_similarity_user(preds: list[int], item_sim: np.ndarray,
                                    k: int) -> float:
    """
    calculate intra list similarity using the history of item interactions
    :param preds: the predictions for a single user
    :param item_sim: the similarity between items
    :param k: the number of items to consider in the predictions
    :return: intra list diversity for a user
    """
    if k == 1 or len(preds) == 1:
        return 1.0
    if len(preds) < k:
        k = len(preds)

    intra_list_similarity = 0
    for i in range(k):
        for j in range(i + 1, k):
            if preds[i] >= item_sim.shape[0] or preds[j] >= item_sim.shape[0]:
                print('out of bounds')
                continue
            intra_list_similarity += item_sim[preds[i], preds[j]]

    if k == 2 or len(preds) == 2:
        return intra_list_similarity

    denominator = (k - 1) * (k - 2) / 2
    return intra_list_similarity / denominator


def get_intra_list_similarity(pred: dict[int, list[int]], ui_mat,
                              k: int) -> float:
    """
    calculate intra list diversity using information about the items
    define an item by its user-item interaction vector
    src: https://dl.acm.org/doi/pdf/10.1145/1060745.1060754 (3.3)
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param ui_mat: user-item interaction matrix
    :param k: the number of items to compare to each other in the list
    :return: the average intra list diversity
    """
    # relevant users
    relevant_users = list(pred.keys())

    # pairwise cosine similarity btwn items
    item_dist = sklearn.metrics.pairwise.cosine_similarity(ui_mat.T)

    intra_list_similarity = 0
    for user in relevant_users:
        if user not in pred:
            continue
        intra_list_similarity += _get_intra_list_similarity_user(pred[user],
                                                                item_dist, k)

    return intra_list_similarity / len(relevant_users)  # average over all users
