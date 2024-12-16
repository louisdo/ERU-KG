import pytrec_eval, json
import numpy as np


def evaluation(qrels, predictions):
    """
    qrels = {
        'q1': {
            'd1': 0,
            'd2': 1,
            'd3': 0,
        },
        'q2': {
            'd2': 1,
            'd3': 1,
        },
    }
    run = {
        'q1': {
            'd1': 1.0,
            'd2': 0.0,
            'd3': 1.5,
        },
        'q2': {
            'd1': 1.5,
            'd2': 0.2,
            'd3': 0.5,
        }
    }
    """

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})

    eval_res =  evaluator.evaluate(predictions)

    with open("test_gitig_.json", "w") as f:
        json.dump([eval_res, predictions, qrels], f, indent = 4)

    res = {}
    for metric in ["map", "ndcg"]:
        all_metrics = [v[metric] for v in eval_res.values()]
        res[metric] = np.mean(all_metrics)

    return res