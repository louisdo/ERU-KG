import json
import pandas as pd

data = []
with open("bm25_eval_results.txt") as f:
    for line in f:
        jline = json.loads(line)
        model_name = jline["name"]

        results = {"model_name": model_name}
        for item in jline["1000"]:
            results.update(item)


        data.append(results)



df = pd.DataFrame(data)
df.sort_values(by=['A', 'B'], ascending=[True, False])

df.to_csv("bm25_eval_results.csv", index = False)