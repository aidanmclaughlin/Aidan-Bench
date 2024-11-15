from openai import OpenAI
import os
from functools import lru_cache
from retry import retry
import azure

if os.getenv("AZURE_API_KEY"):
    print("Using Azure API Key")
else:
    print("Not using Azure API Key")


@retry(backoff=2)
def chat_with_model(prompt, model, max_tokens=4000, temperature=0):
    if "o1" in model.lower() or "gpt-4o-2024-08-06" in model.lower() or "gpt-4-turbo" in model.lower() and os.getenv("AZURE_API_KEY"):
        model = model.replace("openai/", "")
        return azure.message_chat(
            [{"role": "user", "content": prompt}],
            model
        )
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
