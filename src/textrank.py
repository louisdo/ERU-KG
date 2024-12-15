import pke


TEXTRANK = pke.unsupervised.TextRank()


def keyphrase_extraction(doc, top_k = 10, extractor = TEXTRANK):
    extractor.load_document(input = doc, language = "en")
    extractor.grammar_selection(grammar = "NP: {<ADJ>*<NOUN|PROPN>+}")
    extractor.candidate_weighting()

    return {
        "present": extractor.get_n_best(n = top_k, stemming = False),
        "absent": []
    }



if __name__ == "__main__":
    test_text = """In this work, we study the problem of unsupervised open-domain keyphrase generation,
where the objective is a keyphrase generation model that can be built without using
human-labeled data and can perform consistently across domains. To solve this problem,
we propose a seq2seq model that consists of
two modules, namely phraseness and informativeness module, both of which can be built in
an unsupervised and open-domain fashion. The
phraseness module generates phrases, while the
informativeness module guides the generation
towards those that represent the core concepts
of the text. We thoroughly evaluate our proposed method using eight benchmark datasets
from different domains. Results on in-domain
datasets show that our approach achieves stateof-the-art results compared with existing unsupervised models, and overall narrows the gap
between supervised and unsupervised methods down to about 16%. Furthermore, we
demonstrate that our model performs consistently across domains, as it overall surpasses
the baselines on out-of-domain datasets"""
    print(keyphrase_extraction(test_text))