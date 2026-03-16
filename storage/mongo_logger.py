# storage/mongo_logger.py
from datetime import datetime

from storage.mongo_client import get_database


class MongoRunLogger:
    def __init__(self):
        self.db = get_database()
        self.runs = self.db["pipeline_runs"]

    def log_run(self, payload: dict):
        payload = {
            **payload,
            "created_at": datetime.utcnow(),
        }
        return self.runs.insert_one(payload)