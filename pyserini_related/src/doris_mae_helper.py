from tqdm import tqdm
def create_aspect_id2annotation(data):
    """
    Input: data with at least the "Annotation" key
    
    Output: return the aspect_id to annotation dictionary, where the key is the aspect_id
            and the value is a list of annotated question pairs whose aspect_id is equal to
            the aforementioned aspect_id
    """
    aspect_id2annotation= {}
    Annotation = data['Annotation']

    for a in Annotation:
        if a['aspect_id'] not in aspect_id2annotation.keys():
            aspect_id2annotation[a['aspect_id']]=[]
        aspect_id2annotation[a['aspect_id']].append(a)
    return aspect_id2annotation

def compute_relevance_score(abstract_id, query_id, data, aspect_id2annotation, normalization= False):
    """
    input: abstract_id is the id for the paper abstract
           query_id is the id for the query, note it could a complex query or a subquery, 
           for the case of sub_query, the data needs to have a subquery dataset as well, similar to Query
           which contains only the complex query. 
           data needs to have the following keys for dataset, "Annotation", "Query"
     
    output: the relevance score for the paper abstract conditioned on the query. If normalization is true, return
            the normalized score, i.e. the score divided by the number of aspects/sub-aspects
            
    Exception: if abstract not in the query's candidate pool, raise value error. 
    """
    Query = data['Query']
    
    q = Query[query_id]
    query_candidate_pool = Query[query_id]['candidate_pool']
    
    if abstract_id not in query_candidate_pool:
        raise ValueError("Abstract not in the Query Candidate Pool, Cannot Perform Computation")
    else:
        aspect_list = []
        for a in q['aspects'].keys():
            aspect_list.append(a)
            aspect_list += q['aspects'][a]
        aspect_list = list(set(aspect_list))
        
        ret = 0
        for asp_id in aspect_list:
            for pair in aspect_id2annotation[asp_id]:
                if pair['abstract_id'] == abstract_id:
                    ret += pair['score']
        if normalization:
            return ret/len(aspect_list)
        else:
            return ret

def compute_all_relevance_score(data, normalization = False):
    """
    input: data needs to have the following keys for dataset, "Query", "Annotation"
    
    output: return the (normalized) scores for each paper in candidate pool for each query, format
            is list of dictionary, the key of the dictionary is the abstract_id, whose value is the 
            relevance score. 
    """
    Query = data['Query']
    aspect_id2annotation = create_aspect_id2annotation(data)
    
    ret = []
    for query_id in tqdm(range(len(Query))):
        score_dict = {}
        for abstract_id in Query[query_id]['candidate_pool']:
            score_dict[abstract_id] = compute_relevance_score(abstract_id = abstract_id, \
                                                              query_id= query_id, data = data, \
                                                              aspect_id2annotation= aspect_id2annotation,\
                                                             normalization = normalization)
        ret.append(score_dict)
    return ret


def compute_all_gpt_score(data):
    """
    input: data needs to have the following keys for dataset, "Query", "Annotation"
    
    output: return the original and normalized scores for each paper in candidate pool for each query, format
            is list of dictionary, the key of the dictionary is the abstract_id, whose value is the 
            relevance score.
    """
    original_gpt_score = compute_all_relevance_score(data, normalization = False)
    normalized_gpt_score = compute_all_relevance_score(data, normalization = True)
    
    result_dict = []
    for ind in range(len(original_gpt_score)):
        temp_dict = {}
        for abs_id in original_gpt_score[ind].keys():
            temp_dict[abs_id] = (original_gpt_score[ind][abs_id], normalized_gpt_score[ind][abs_id])
        result_dict.append(temp_dict)
    return result_dict