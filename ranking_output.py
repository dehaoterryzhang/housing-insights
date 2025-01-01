from numpy import dot
from numpy.linalg import norm
from openai import AzureOpenAI
import os

# Access the API key
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key = api_key,
    api_version= api_version,
    azure_endpoint= api_base
)

def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))

def rank_results(query_embedding, documents):
    """Rank documents based on similarity to the query embedding."""
    return sorted(
        documents,
        key=lambda doc: cosine_similarity(query_embedding, doc['embedding']),
        reverse=True
    )

def generate_response(query, context, model="gpt-4o"):
    """Generate a response using GPT."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a housing insights assistant."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}"}
        ]
    )
    return response.choices[0].message.content
