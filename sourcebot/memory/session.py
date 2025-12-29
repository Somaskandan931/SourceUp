import redis, json
r = redis.Redis(decode_responses=True)

def get(sid): return json.loads(r.get(sid) or "{}")
def set_(sid,d): r.set(sid,json.dumps(d))
