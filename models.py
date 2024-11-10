from openai import OpenAI
import os
from functools import lru_cache
from retry import retry


@retry(tries=3)
def chat_with_model(prompt, model, max_tokens=4000, temperature=0):
    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    extra_body = {}
    if model == "meta-llama/llama-3.1-405b-instruct:bf16":
        model = "meta-llama/llama-3.1-405b-instruct"
        extra_body = {
            "provider": {
                "quantizations": ["bf16"]
            }
        }
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body
    )
    return response.choices[0].message.content


@lru_cache(maxsize=10000)
@retry(tries=3)
def embed(text):
    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-large", input=[text])
    return response.data[0].embedding
