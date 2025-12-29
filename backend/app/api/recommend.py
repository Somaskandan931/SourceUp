from fastapi import APIRouter
from backend.app.models.retriever import retrieve
from backend.app.models.ranker import score
from backend.app.services.explanation import explain

router = APIRouter()

@router.post("/recommend")
def recommend(q:dict):
    res=[]
    for s in retrieve(q["product"]):
        res.append({
            "supplier":s["supplier_name"],
            "product":s["product_name"],
            "score":score(s,q),
            "reasons":explain(s,q)
        })
    return sorted(res,key=lambda x:x["score"],reverse=True)
