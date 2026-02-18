from flask import Flask, request, jsonify
import time, hashlib
import numpy as np


app = Flask(__name__)

cache = {}
MAX_CACHE = 1500
TTL = 86400

stats = {
    "hits":0,
    "miss":0
}

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def normalize(q):
    return q.strip().lower()


def get_hash(q):
    return hashlib.md5(q.encode()).hexdigest()

def embed(text):
    vec = [ord(c) for c in text[:50]]
    return np.array(vec + [0]*(50-len(vec)))

def semantic_search(vec):
    for key,data in cache.items():
        sim = cosine_similarity(vec, data["embedding"])
        if sim > 0.95:
            return key,data
    return None,None


def call_ai(q):
    time.sleep(1)
    return "Moderation result: SAFE"

def evict():
    if len(cache) >= MAX_CACHE:
        oldest = min(cache.items(), key=lambda x: x[1]["time"])[0]
        del cache[oldest]

@app.route("/", methods=["POST"])
def query():
    start = time.time()
    q = request.json["query"]
    qn = normalize(q)
    key = get_hash(qn)

    # TTL check
    if key in cache and time.time()-cache[key]["time"] > TTL:
        del cache[key]

    # Exact match
    if key in cache:
        stats["hits"]+=1
        latency = int((time.time()-start)*1000)
        return jsonify(answer=cache[key]["answer"],cached=True,latency=latency,cacheKey=key)

    # Semantic match
    vec = embed(qn)
    k,data = semantic_search(vec)
    if data:
        stats["hits"]+=1
        latency = int((time.time()-start)*1000)
        return jsonify(answer=data["answer"],cached=True,latency=latency,cacheKey=k)

    # MISS → call AI
    stats["miss"]+=1
    ans = call_ai(qn)

    evict()
    cache[key] = {"answer":ans,"time":time.time(),"embedding":vec}

    latency = int((time.time()-start)*1000)
    return jsonify(answer=ans,cached=False,latency=latency,cacheKey=key)


@app.route("/analytics")
def analytics():
    total = stats["hits"]+stats["miss"]
    hitrate = stats["hits"]/total if total else 0
    cost_saved = stats["hits"] * 300 * 0.40 / 1_000_000

    return jsonify(
        hitRate=round(hitrate,2),
        totalRequests=total,
        cacheHits=stats["hits"],
        cacheMisses=stats["miss"],
        cacheSize=len(cache),
        costSavings=round(cost_saved,2),
        savingsPercent=round(hitrate*100,2),
        strategies=["exact match","semantic similarity","LRU eviction","TTL expiration"]
    )

import os
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

