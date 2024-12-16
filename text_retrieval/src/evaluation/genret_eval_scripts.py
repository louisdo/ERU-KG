import os
import numpy as np

def eval_all(predict, label):
    log_dict = {}
    log_dict.update(eval_recall(predict, label, at=1))
    log_dict.update(eval_recall(predict, label, at=10))
    # log_dict.update(eval_recall(predict, label, at=20))
    # log_dict.update(eval_recall(predict, label, at=50))
    log_dict.update(eval_recall(predict, label, at=100))
    # log_dict.update(eval_recall(predict, label, at=1000))
    log_dict.update(eval_mrr(predict, label, at=10))
    log_dict.update(eval_mrr(predict, label, at=100))
    # log_dict.update(eval_ndcg(predict, label, at=10))
    log_dict.update(eval_ndcg_rank(predict, label, at=10))
    return log_dict


def base_it(predict, label, at, score_func):
    assert len(predict) == len(label)
    scores = []
    for pred, lbs in zip(predict, label):
        pred = pred.tolist() if not isinstance(pred, list) else pred
        best_score = 0.
        if not isinstance(lbs, list):
            lbs = [lbs]
        for lb in lbs:
            if isinstance(lb, list):
                lb = lb[0]
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            best_score = max(best_score, cur_score)
        scores.append(best_score)
    return scores


def eval_recall(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: int(rank != 0))
    return {f'R@{at}': sum(scores) / len(scores)}


def eval_mrr(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: 1 / rank if rank != 0 else 0)
    return {f'MRR@{at}': sum(scores) / len(scores)}


def eval_ndcg(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: 1 / np.log2(rank + 1) if rank != 0 else 0)
    return {f'nDCG@{at}': sum(scores) / len(scores)}


def eval_ndcg_rank(predict, label, at=10):
    assert len(predict) == len(label)
    scores = []
    score_func = lambda rk: 1 / np.log2(rk + 1) if rk != 0 else 0
    idcg_set = [score_func(i + 1) for i in range(10)]
    for pred, lbs in zip(predict, label):
        if not isinstance(lbs, list):
            lbs = [lbs]

        if len(lbs) == 0:
            continue
        pred = pred.tolist() if not isinstance(pred, list) else pred
        dcg = 0.

        if isinstance(lbs[0], list):
            lbs.sort(key=lambda x: x[1], reverse=True)

        for lb in lbs:
            if isinstance(lb, list):
                lb, rel = lb
            else:
                rel = 1
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            dcg += cur_score * rel
        if isinstance(lbs[0], list):
            idcg = sum([x * y[1] for x, y in zip(idcg_set[:len(lbs)], lbs)])
        else:
            idcg = sum(idcg_set[:len(lbs)])
        best_score = dcg / idcg
        scores.append(best_score)

    return {f'nDCG@{at}': sum(scores) / len(scores)}


if __name__ == "__main__":
    print(eval_recall(predict=[[1,2,3,5]], label = [5], at = 10))