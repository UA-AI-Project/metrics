def _get_calibrated_recall_user(preds: list[int], truths: list[int],
                                k: int) -> float:
    # depending on the list that is empty, the recall is 0 or 1
    if not len(truths):
        return 1.0
    elif not len(preds):
        return 0.0

    truths_set = set(truths)
    if len(preds) < k:
        k = len(preds)

    recall = 0
    for i in range(k):
        if preds[i] in truths_set:
            recall += 1
            truths_set.remove(preds[i])

    return recall / min(len(truths), k)


def get_calibrated_recall(pred: dict[int, list[int]],
                          truth: dict[int, list[int]], k: int) -> float:
    """
    Compute the recall of a given prediction.
    src: https://recpack.froomle.ai/generated/recpack.metrics.CalibratedRecallK.html#recpack.metrics.CalibratedRecallK
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param truth: list of ground truth items per user
    :param k: number of items to consider for the recall
    :return: recall@k
    """
    # if nothing needs to be recalled the recall is 1
    if k == 0:
        return 1.0

    total_recall = 0
    for user in pred.keys():
        if user not in truth:
            continue
        total_recall += _get_calibrated_recall_user(pred[user], truth[user], k)

    # beware of dividing it by all the users
    return total_recall / len(truth.keys())
