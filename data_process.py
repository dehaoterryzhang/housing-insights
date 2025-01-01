import os
import requests
import pandas as pd

DATA_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1735349373"

# Create a directory to store the dataset
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# File path to save the dataset
FILE_PATH = os.path.join(DATA_DIR, "zillow_data.csv")

# Download the dataset
def download_dataset(url, file_path):
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Dataset saved to {file_path}")
    else:
        print(f"Failed to download the dataset. HTTP Status Code: {response.status_code}")

# Run the download function
download_dataset(DATA_URL, FILE_PATH)

# Load the dataset
def load_dataset(file_path):
    try:
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Load the data
df = load_dataset(FILE_PATH)

# Step 1: Inspect column names
# Identify columns that represent dates (e.g., those starting from '2000-01')
date_columns = [col for col in df.columns if col[:4].isdigit()]

# Step 2: Filter columns to include only January 2022 onward
filtered_columns = [col for col in date_columns if col >= "2022-01"]

# Keep only relevant columns: RegionName (ZIP Code), StateName, and filtered date columns
df = df[["RegionName", "City", "StateName"] + filtered_columns]

# Step 3: Melt the data into long format
# This converts wide format (many date columns) into long format (one date per row)
df_long = df.melt(
    id_vars=["RegionName", "City", "StateName"],  # Columns to keep as-is
    var_name="Date",                     # New column name for the melted column headers (dates)
    value_name="HomeValue"               # New column name for the melted values (home values)
)

# Step 4: Drop rows with missing values
df_long.dropna(subset=["HomeValue"], inplace=True)

# Step 5: Convert the "Date" column to a datetime object for easier filtering
df_long["Date"] = pd.to_datetime(df_long["Date"])

# Step 6: Optional: Add textual representation for RAG (can be used later for embeddings)
df_long["Text"] = df_long.apply(
    lambda row: f"The home value in zipcode {row['RegionName']} (City name: {row['City']}, State code: {row['StateName']}) in {row['Date'].strftime('%B %Y')} was ${row['HomeValue']:.2f}.",
    axis=1
)

df_long.to_csv("data/zillow_data_reformatted.csv", index=False)
print("Reformatted dataset saved to 'data/zillow_data_reformatted.csv'")