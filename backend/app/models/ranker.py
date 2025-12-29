def score(s, q):
    score = 0
    if s["price"] <= q["max_price"]: score += 0.4
    if s["location"] == q["location"]: score += 0.2
    if q["certification"] in s["certifications"]: score += 0.3
    score += (s["rating"]/5)*0.1
    return round(score,3)
