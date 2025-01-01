from openai import AzureOpenAI
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key = api_key,
    api_version= api_version,
    azure_endpoint= api_base
)

FILE_PATH = "data/zillow_data_reformatted.csv"
df = pd.read_csv(FILE_PATH)

def get_embedding(text, model):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text}")
        print(f"Exception: {e}")
        return None


# Choosing WA as the sample
df = df.loc[df.StateName == 'WA']

print("Generating embeddings...")
df["Embedding"] = df["Text"].apply(lambda x: get_embedding(x, model="text-embedding-ada-002-dec24"))

# Save the embeddings
embeddings = np.array(df["Embedding"].tolist())
np.save("data/zillow_embeddings.npy", embeddings)

# Save metadata (RegionName, StateName, Date, Text) for indexing
metadata = df[["RegionName", "StateName", "City", "Date", "Text"]]
metadata.to_csv("data/zillow_metadata.csv", index=False)

print("Embeddings and metadata saved successfully!")