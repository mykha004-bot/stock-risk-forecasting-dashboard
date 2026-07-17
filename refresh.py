"""Incremental refresh: pull only new dates per ticker.

    python refresh.py            # incremental
    python refresh.py --full     # force full re-pull
"""
import argparse
import logging
import config
import data_pipeline as dp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="force full re-pull")
    args = ap.parse_args()
    res = dp.refresh(full=args.full)
    print("\n=== Refresh summary ===")
    print(res.to_string(index=False))
    print(f"\nDB: {config.DB_PATH}")
