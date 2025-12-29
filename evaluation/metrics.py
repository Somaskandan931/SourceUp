def precision_at_k(res,rel,k=5):
    return len(set(res[:k])&set(rel))/k
