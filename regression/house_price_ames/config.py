import os
from dotenv import load_dotenv

load_dotenv()  # reads .env

# Defaults if not provided in .env
SEED = int(os.getenv("SEED", 42))
RAW_DIR = os.getenv("RAW_DIR", "data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "data/snapshots")