import httpx

OLLAYA_URL = "http://127.0.0.1:11435/api/decide"

def ollaya_router(query: str):
    """Routes a query through Ollaya's laya decision model; this one is a custom version of it (soma-router) created via a ModelFile using ollaya create"""
    response = httpx.post(
        OLLAYA_URL,
        json={
            "model": "soma-router", # custom distilled model
            "state": query
        }
    )
    decision = response.json()['answers']['answer']['choice']
    return decision

if __name__ == "__main__":
    query = input("Ask: ")
    result = ollaya_router(query)
    print(result)
