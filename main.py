from data_process import download_dataset, load_dataset
from embedding_generator import get_embedding
from index_upload import *
from ranking_output import rank_results, generate_response

if __name__ == "__main__":
    # Data Processing
    raw_data = download_dataset("path/to/file")
    processed_data = load_dataset(raw_data)

    # Embedding Generation
    texts = [doc['content'] for doc in processed_data]
    embeddings = get_embedding(texts)

    # Indexing and Uploading
    documents = [{"content": text, "embedding": embedding} for text, embedding in zip(texts, embeddings)]
    #create_index("your_service_name", "your_api_key", "your_index_name", {...})
    #upload_documents("your_service_name", "your_api_key", "your_index_name", documents)

    # Ranking and Model Output
    query_embedding = get_embedding(["Your search query"])[0]
    ranked_docs = rank_results(query_embedding, documents)
    context = "\n".join([doc['content'] for doc in ranked_docs[:3]])
    response = generate_response("Your search query", context)
    print(response)
