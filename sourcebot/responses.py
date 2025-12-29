def format(res):
    return "\n".join(
        f"{r['supplier']} → {', '.join(r['reasons'])}"
        for r in res[:3]
    )
