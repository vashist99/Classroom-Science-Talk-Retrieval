import os
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
PROMPT_VERSION = "v1"

# (connect, read) seconds. Without this a stalled endpoint blocks the call
# forever; with it a hung request raises and the caller can retry / rotate keys.
REQUEST_TIMEOUT = (10, 180)

def _make_cache_key(prompt_version: str, model: str, endpoint: str, params: dict) -> str:
    payload = {
        "prompt_version": prompt_version,
        "model": model,
        "endpoint": endpoint,
        "params": params
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"

def _load_from_cache(cache_key: str) -> dict | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_to_cache(cache_key: str, response: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_key)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2,sort_keys=True)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise e

def cached_request(
    *,
    api_key: str,
    url: str,
    endpoint: str,         # "completion" or "embedding"
    model: str,
    params: dict,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    cache_key  = _make_cache_key(prompt_version, model, endpoint, params)

    hit = _load_from_cache(cache_key)
    if hit is not None:
        return hit["raw_response"]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, json=params, timeout=REQUEST_TIMEOUT)
    raw = response.json()

    entry = {
        "cache_key": cache_key,
        "prompt_version": prompt_version,
        "model": model,
        "endpoint": endpoint,
        "params": params,
        "raw_response": raw,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    _save_to_cache(cache_key, entry)
    return raw

def connect_to_llm(api_key, model_type):
    completion_url = os.getenv("COMPLETION_URL")
    embedding_url = os.getenv("EMBEDDING_URL")
    if model_type == "augment":
        model = os.getenv("LLM_MODEL_AUGMENT")
    elif model_type == "tracka":
        model = os.getenv("LLM_MODEL_TRACKA")
    elif model_type == "trackb":
        model = os.getenv("LLM_MODEL_TRACKB")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    body_completion = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a 2 line poem about a cat."
            }
        ]
    }
    body_embedding = {
        "model": model, 
        "input": "Write a 2 line poem about a cat."
    }
    response_completion = requests.post(completion_url, headers=headers,
                                        json=body_completion, timeout=REQUEST_TIMEOUT)
    response_embedding = requests.post(embedding_url, headers=headers,
                                       json=body_embedding, timeout=REQUEST_TIMEOUT)
    return response_completion.json(), response_embedding.json()

if __name__ == "__main__":
    api_key = os.getenv("LLM_API_KEY")
    model_type = "augment"
    response_completion, response_embedding = connect_to_llm(api_key, model_type)
    print(response_completion)
    print(response_embedding)