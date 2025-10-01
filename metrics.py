import numpy as np
import pandas as pd


def gini_index(x: np.array):
    """
    src: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797
    :param x: a numpy array of the values
    :return: gini index of array of values
    """
    n = len(x)
    if n <= 1:
        return 0.0
    sorted_x = np.sort(x)
    x_cum = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(x_cum) / x_cum[-1]) / n


def get_top_k(predictions: pd.DataFrame, k: int):
    """
    Processes recommendations so only the first k unique recommendation per user are retained.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    return (
        predictions.drop_duplicates(["user_id", "item_id"])
        .groupby("user_id")
        .head(k)
        .reset_index(drop=True)
    )


def calculate_user_coverage(predictions: pd.DataFrame, k: int, n_users: int) -> float:
    """
    Calculates the fraction of users who received at least k unique recommendations.

    This metric is maximally 1 when at least k items are recommended for each user.
    This metric is minimally 0 when no recommendations are given.
    For fair recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    n_users: int
        Target number of users.
        Must be at least 1.
    """

    if n_users < 1:
        raise ValueError("n_users mut be at least 1.")

    recommendations = get_top_k(predictions, k)
    recommendations_per_user = recommendations["user_id"].value_counts()
    return (recommendations_per_user >= k).sum() / n_users


def calculate_item_coverage(predictions: pd.DataFrame, k: int, n_items: int) -> float:
    """
    Calculates the fraction of items that received at least one recommendation.

    This metric is maximally 1 when each item is recommended at least once.
    This metric is minimally 1/n_items when all recommendations are the same item.
    For diverse and fair recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    n_items: int
        Number of unique items that are recommendable.
        Must be at least 1.
    """

    if n_items < 1:
        raise ValueError("n_items must be at least 1")

    recommendations = get_top_k(predictions, k)
    return recommendations["item_id"].nunique() / n_items


def calculate_item_gini(predictions: pd.DataFrame, k: int) -> float:
    """
    Calculates the gini index of the item distribution.

    This metric is maximally 1 - 1/n_items when all recommendations are the same item.
    This metric is minimally 0 when every item is recommended in equal amounts.
    For fair recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    item_counts = recommendations["item_id"].value_counts()
    return gini_index(item_counts.to_numpy())


def calculate_publisher_gini(
    predictions: pd.DataFrame, k: int, item_mapper: pd.Series
) -> float:
    """
    Calculates the gini index of the publisher distribution.

    This metric is maximally 1 - 1/n_publishers when all recommendations are the same publisher.
    This metric is minimally 0 when every publisher is recommended in equal amounts.
    For fair recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    item_mapper: pandas.Series
        Mapping from item identifiers to publishers.
        The index contains item identifiers, and the values are publisher identifiers (e.g., strings).
        Must have an entry for each item occurring in predictions.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    publisher_counts = item_mapper[recommendations["item_id"]].value_counts()
    return gini_index(publisher_counts.to_numpy())


def calculate_calibrated_recall(
    predictions: pd.DataFrame, k: int, test_out: pd.DataFrame
) -> float:
    """
    Calculates the average calibrated recall@k across all test users.

    The calibrated recall for a user is the number of relevant recommendations (hits) divided by the minimum of k and the number of relevant items for that user.
    The calibrated recall differs from the standard recall because it takes into account that you cannot have more than k hits.

    This metric is maximally 1 when for each user either all recommendation are relevant or all relevant items are among the recommendations.
    This metric is minimally 0 when none of the recommendations are relevant.
    For accurate recommendations, higher is better.

    See https://recpack.froomle.ai/generated/recpack.metrics.CalibratedRecallK.html for more information.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_out: pandas.DataFrame
        A dataframe of ground truth user–item interactions.
        Must contain at least one interaction.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    Notes
    -----
    Users that are not in test_out are ignored.
    """
    if len(test_out) == 0:
        raise ValueError("test_out must contain at least one interaction.")

    recommendations = get_top_k(predictions, k)

    # count the relevant recommendation per user
    hits = pd.merge(recommendations, test_out, on=["user_id", "item_id"])
    hits_per_user = hits["user_id"].value_counts()

    # count the relevant items per user
    relevant_per_user = test_out["user_id"].value_counts()
    calibrated_relevant_per_user = np.minimum(relevant_per_user, k)

    # hits_per_user.user_id is subset of calibrated_relevant_per_user.user_id
    # fill_value=0 ensures we handle the missing users
    return hits_per_user.div(calibrated_relevant_per_user, fill_value=0).mean()


def calculate_ndcg(predictions: pd.DataFrame, k: int, test_out: pd.DataFrame) -> float:
    """
    Calculates the average NDCG@k across all test users.

    The NDCG for a user is a normalized weighted sum of relevant recommendations (hits).
    The weights are determined by the rank of the recommendation (with earlier recommendation having more weight).
    The normalization is determined so that k hits results in an NDCG of 1.

    This metric is maximally 1 when for each user either all recommendation are relevant or all relevant items are among the first recommendations.
    This metric is minimally 0 when none of the recommendations are relevant.
    For accurate recommendations, higher is better.

    See https://recpack.froomle.ai/generated/recpack.metrics.NDCGK.html for more information.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_out: pandas.DataFrane
        A dataframe of ground truth user–item interactions.
        Must contain at least one interactions
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    Notes
    -----
    Users that are not in test_out are ignored.
    """
    if len(test_out) == 0:
        raise ValueError("test_out must contain at least one interaction.")

    recommendations = get_top_k(predictions, k)

    # calculate dcg = weighted sum of hits per user
    recommendations["weight"] = 1 / np.log2(
        recommendations.groupby("user_id").cumcount() + 2
    )
    hits = pd.merge(recommendations, test_out, on=["user_id", "item_id"])
    dcg_per_user = hits.groupby("user_id")["weight"].sum()

    # calculate ideal dcg per user
    idcg_table = np.cumsum(1 / np.log2(np.arange(k) + 2))
    relevant_counts = test_out["user_id"].value_counts()
    calibrated_relevant_counts = np.minimum(relevant_counts, k)
    idcg_per_user = pd.Series(
        idcg_table[calibrated_relevant_counts - 1], calibrated_relevant_counts.index
    )

    # dcg.user_id is subset of idcg.user_id
    # fill_value=0 ensures we handle the missing users
    return dcg_per_user.div(idcg_per_user, fill_value=0).mean()


def calculate_intra_list_sim(
    predictions: pd.DataFrame,
    k: int,
    item_similarity_matrix: np.ndarray,
) -> float:
    """
    Calculates the average intra list similarity across users [1].

    The intra list similarity for a user is the average pairwise similarity of their recommendations.

    This metric is maximally 1 when for each user, all recommended items are fully similar (similarity=1).
    This metric is minimally 0 when for each user, none of the recommendations are similar (similarity=0).
    For diverse recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    item_similarity_matrix: np.ndarray of shape (n_items, n_items)
        A matrix containing pairwise items similarities between 0 and 1.
        Must have entries for all items in predictions and test_in.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.

    References
    ----------
    .. [1] Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen. 2005.
           Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web (WWW '05). Association for Computing Machinery, New York, NY, USA, 22–32.
           https://doi.org/10.1145/1060745.1060754
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    # gather all pairwise similarities
    cross = pd.merge(recommendations, recommendations, on="user_id")
    cross = cross[cross["item_id_x"] < cross["item_id_y"]]
    cross["value"] = item_similarity_matrix[cross["item_id_x"], cross["item_id_y"]]

    # compute average pairwise similarity  per user
    ils_per_user = cross.groupby("user_id")["value"].mean()

    # average ils across users
    return ils_per_user.mean()


def calculate_novelty(
    predictions: pd.DataFrame,
    k: int,
    test_in: pd.DataFrame,
    item_distance_matrix: np.ndarray,
) -> float:
    """
    Calculates the average novelty across users [1].

    The novelty for a user is the average distance between their recommendations and their history.

    This metric is maximally 1 when for each test user, all recommended items are dissimilar to their history (distance=1).
    This metric is minimally 0 when for each test user, all recommended items are similar to their history (distance=0).
    For diverse recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_in: pandas.DataFrame
        A dataframe of historical user–item interactions (fold-in).
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    item_distance_matrix: np.ndarray of shape (n_items, n_items)
        A matrix containing pairwise items distances between 0 and 1.
        Must have entries for all items in predictions and test_in.

    Notes
    -----
    If no valid recommendations are provided, the result is 0.

    References
    ----------
    .. [1] Hurley, N. and Zhang, M. 2011.
           Novelty and Diversity in Top-N Recommendation – Analysis and Evaluation. ACM Trans. Internet Technol. 10, 4, Article 14 (March 2011), 30 pages.
           https://doi.org/10.1145/1944339.1944341
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 0

    # gather all pairwise similarities between predictions and history
    cross = pd.merge(recommendations, test_in, on="user_id")
    cross["value"] = item_distance_matrix[cross["item_id_x"], cross["item_id_y"]]

    # compute average similarity per user
    novelty_per_user = cross.groupby("user_id")["value"].mean()

    # average novelty across users
    return novelty_per_user.mean()
