import argparse
import sys
import os
from datetime import datetime, timedelta, UTC
from sqlmodel import Session, create_engine, select

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import InsightRecord
from settings import settings

def prune_old_records(days: int):
    engine = create_engine(f"sqlite:///{settings.db_path}")
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

    with Session(engine) as session:
        statement = select(InsightRecord).where(InsightRecord.processed_at < cutoff)
        old_records = session.exec(statement).all()
        
        for record in old_records:
            session.delete(record)
        
        session.commit()
        print(f"Pruned {len(old_records)} records older than {days} days.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune old records from the SQLite database.")
    parser.add_argument("--days", type=int, default=90, help="Number of days of records to keep (default: 90)")
    args = parser.parse_args()
    
    prune_old_records(args.days)
