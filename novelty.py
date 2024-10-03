import numpy as np
import sklearn.metrics
from scipy.sparse import csr_matrix


def get_novelty(pred: dict[int, list[int]], ui_mat: csr_matrix, k: int) -> float:
    """
    src: https://dl.acm.org/doi/pdf/10.1145/1944339.1944341 (3.2)
    we use cosine distance as the distance measure
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param ui_mat: the user item interaction matrix
    :param k: the number of items to consider in the predictions
    :param full_ui_mat: the full user item interaction matrix
    :return: the average novelty
    """
    if k == 0:
        return 0.0

    # pairwise cosine distance btwn items, using only training interactions
    item_dist = sklearn.metrics.pairwise.cosine_distances(ui_mat.T)

    total_novelty = 0.0
    for u, items in pred.items():
        # limit the items interacted with to the top-k
        top_k_items = items[:k]

        # L contains the items that a user has interacted with
        # get nonzero indices of the user-item interaction matrix for user u
        L = np.nonzero(ui_mat[u])[1]

        user_novelty = 0.0
        cnt = 0
        for pred_item in top_k_items:
            for all_item in L:
                if all_item == pred_item:
                    continue

                cnt += 1
                item_novelty = item_dist[pred_item, all_item]
                user_novelty += item_novelty

        # add the user's novelty to the total novelty
        total_novelty += user_novelty / max(cnt, 1)

    return total_novelty / len(pred.keys())  # average over all users
