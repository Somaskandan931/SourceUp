import re

def rule_parse(t):
    return {
        "product": t,
        "max_price": float(re.search(r"\d+",t).group()) if re.search(r"\d+",t) else 1e9,
        "location": "chennai" if "chennai" in t.lower() else "",
        "certification": "iso" if "iso" in t.lower() else ""
    }
