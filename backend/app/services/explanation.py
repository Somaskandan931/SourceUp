def explain(s,q):
    r=[]
    if s["price"]<=q["max_price"]: r.append("Within budget")
    if s["location"]==q["location"]: r.append("Local supplier")
    if q["certification"] in s["certifications"]: r.append("Certification match")
    return r
