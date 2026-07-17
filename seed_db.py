"""One-time (or occasional) full-history seed. Run locally, then commit data/prices.db.

    python seed_db.py
"""
import logging
import config
import data_pipeline as dp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    res = dp.seed()
    print("\n=== Seed summary ===")
    print(res.to_string(index=False))
    ok = (res["status"] == "ok").sum()
    print(f"\n{ok}/{len(res)} tickers seeded OK -> {config.DB_PATH}")
    failed = res[res["status"] != "ok"]["ticker"].tolist()
    if failed:
        print(f"Not seeded (re-run to retry): {failed}")
