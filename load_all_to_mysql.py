import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Database credentials from .env
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "swiggy_db")

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(MYSQL_URL)

csv_files = [
    "dim_campaign.csv",
    "dim_coupon.csv",
    "dim_customer.csv",
    "dim_geo.csv",
    "dim_restaurant_product.csv",
    "fact_transactions.csv"
]

for file_name in csv_files:
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not found. Skipping...")
        continue

    table_name = os.path.splitext(file_name)[0]
    print(f"Loading {file_name} -> MySQL table `{table_name}`...")
    
    # Read CSV
    df = pd.read_csv(file_name)
    
    # Standardize column names (lowercase, remove trailing spaces)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ingest into MySQL (replaces table if already existing)
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False, chunksize=2000)
    print(f"Successfully loaded `{table_name}` ({len(df)} rows).")

print("\nAll 6 tables are now live in MySQL 'swiggy_db'!")
