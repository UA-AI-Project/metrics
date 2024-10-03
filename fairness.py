from collections import Counter

from .gini_index import calc_gini_index


def get_publisher_fairness(pred: dict, i2p_dict: dict, k: int) -> float:
    """
    work with publisher fairness, so that each publisher has the same
    representation in the top-k recommendations
    use gini index to measure the fairness
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param k: the number of items to consider in the predictions
    :return: the gini index grouped by publishers
    """
    if k == 0:
        return 0.0

    # group the predictions per publisher
    pred_per_publisher = Counter()
    for user in pred.keys():
        preds = pred[user][:min(k, len(pred[user]))]
        if not len(preds):
            continue

        publisher_preds = map(lambda i: i2p_dict[i], preds)
        pred_per_publisher += Counter(publisher_preds)

    # calculate the gini index
    gini = calc_gini_index(list(pred_per_publisher.values()))

    return gini
