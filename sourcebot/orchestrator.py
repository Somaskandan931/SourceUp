import requests
from memory.session import get,set_
from nlu.parser import parse

API="http://localhost:8000/recommend"

def handle(sid,text):
    s=get(sid)
    s.update(parse(text))
    set_(sid,s)

    if not s.get("location"):
        return "Which location?"

    return requests.post(API,json=s).json()
