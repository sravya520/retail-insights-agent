"""Export the sales and returns tables to parquet in data/deploy/ (committed, so a free host can rebuild the database).

Run after python -I -m ria.sql_db:   python scripts/export_deploy_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ria.sql_db import export_parquet  # noqa: E402

if __name__ == "__main__":
    sizes = export_parquet()
    for table, size in sizes.items():
        print(f"{table}.parquet: {size / 1e6:.1f} MB")
