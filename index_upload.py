from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex,SearchField,SimpleField,SearchableField,VectorSearch,VectorSearchAlgorithmConfiguration,SearchFieldDataType,ComplexField,VectorSearchAlgorithmKind
from azure.core.credentials import AzureKeyCredential
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import uuid

# Load environment variables from the .env file
load_dotenv()

# Access the Search API key
endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
index_name = "housing-insights-index"
api_key = os.environ["AZURE_SEARCH_API_KEY"]

index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

# Define the vector search algorithm configuration
vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(name="default", kind=VectorSearchAlgorithmKind.HNSW)
    ]
)

# issue with the vector search configuration
# Define the fields
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
    SearchableField(name="RegionName", type=SearchFieldDataType.String, filterable=True, sortable=True),
    SearchableField(name="StateName", type=SearchFieldDataType.String, filterable=True, sortable=True),
    SearchableField(name="City", type=SearchFieldDataType.String, filterable=True, sortable=True),
    SimpleField(name="Date", type=SearchFieldDataType.String, filterable=True, sortable=True),
    SearchableField(name="Text", type=SearchFieldDataType.String),
    SimpleField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector=True,
        searchable=True,
        dimensions=1536,
        vector_search_configuration="default"
    )
]


# Create the index
index = SearchIndex(name=index_name, fields=fields)
index_client.create_index(index)
print(f"Index '{index_name}' created successfully!")

# Load metadata CSV
metadata_file = "data/zillow_metadata.csv"
metadata_df = pd.read_csv(metadata_file)

# Generate unique IDs for each row
#metadata_df['id'] = [str(uuid.uuid4()) for _ in range(len(metadata_df))]

# Save the updated metadata with IDs
#metadata_df.to_csv("data/zillow_metadata.csv", index=False)

# Load embeddings
embeddings_file = "data/zillow_embeddings.npy"
embeddings = np.load(embeddings_file)

# Combine IDs and embeddings into a structured format for upload
data_to_upload = [
    {
        "id": metadata_df.iloc[i]["id"],
        "RegionName": metadata_df.iloc[i]["RegionName"],
        "StateName": metadata_df.iloc[i]["StateName"],
        "City": metadata_df.iloc[i]["City"],
        "Date": metadata_df.iloc[i]["Date"],
        "Text": metadata_df.iloc[i]["Text"],
        "embedding": embeddings[i].tolist()
    }
    for i in range(len(metadata_df))
]

# Upload Documents in Batches
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
batch_size = 1000  # Azure Search recommends uploading in batches
for i in range(0, len(data_to_upload), batch_size):
    batch = data_to_upload[i:i+batch_size]
    result = client.upload_documents(documents=batch)
    print(f"Uploaded batch {i // batch_size + 1}: {result}")

