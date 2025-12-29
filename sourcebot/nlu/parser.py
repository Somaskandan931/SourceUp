import json
from nlu.rules import rule_parse
from nlu.gpt_fallback import gpt_parse

def parse(text):
    r = rule_parse(text)
    if not r["product"]:
        try: r = json.loads(gpt_parse(text))
        except: pass
    return r
