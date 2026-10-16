import os
import shutil
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
FACT_CSV = "fact_transactions.csv"
REST_CSV = "dim_restaurant_product.csv"

# 1. Clean previous database directory if it exists
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print(f"Removed previous vector store at {CHROMA_DIR}")

# 2. Read datasets
print("Reading fact and dimension data...")
fact_df = pd.read_csv(FACT_CSV)
rest_df = pd.read_csv(REST_CSV)

# 3. Filter rows containing customer text reviews
feedback_df = fact_df[fact_df["feedback_text"].notnull()].copy()
print(f"Total reviews found: {len(feedback_df)}")

# Sample 6,000 reviews for fast local indexing and querying
sample_size = min(6000, len(feedback_df))
feedback_sample = feedback_df.sample(n=sample_size, random_state=42)

# 4. Enrich feedback with restaurant and cuisine metadata
merged_df = feedback_sample.merge(
    rest_df[["restaurant_product_id", "restaurant_name", "product_name", "cuisine_tag"]],
    on="restaurant_product_id",
    how="left"
)

# 5. Build enriched content for semantic search
merged_df["content_to_embed"] = (
    "Restaurant: " + merged_df["restaurant_name"].fillna("Unknown") +
    " | Cuisine: " + merged_df["cuisine_tag"].fillna("General") +
    " | Product: " + merged_df["product_name"].fillna("Item") +
    " | Rating: " + merged_df["rating"].astype(str) +
    " | City: " + merged_df["city"].fillna("Unknown") +
    " | Feedback: " + merged_df["feedback_text"]
)

# Keep metadata columns for structured filtering
keep_cols = ["content_to_embed", "restaurant_name", "cuisine_tag", "city", "rating", "delivery_minutes"]
final_df = merged_df[keep_cols].copy()

# 6. Convert DataFrame rows into LangChain Document objects
loader = DataFrameLoader(final_df, page_content_column="content_to_embed")
docs = loader.load()

# 7. Generate embeddings using free local sentence-transformers
print("Generating embeddings using sentence-transformers/all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 8. Create and persist the vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"\nSuccess! Indexed {len(docs)} enriched feedback records into '{CHROMA_DIR}'.")