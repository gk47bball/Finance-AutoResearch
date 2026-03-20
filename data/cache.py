"""SQLite cache with TTL expiry for all data fetches."""

import sqlite3
import time
import pickle
import os
from typing import Any, Optional


class CacheDB:
    def __init__(self, db_path: str = "cache/autoresearch.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    def get(self, key: str, ttl_hours: float) -> Optional[Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, created_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value_blob, created_at = row
            if time.time() - created_at > ttl_hours * 3600:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            return pickle.loads(value_blob)

    def put(self, key: str, value: Any):
        blob = pickle.dumps(value)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, blob, time.time()),
            )

    def invalidate(self, key_prefix: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key LIKE ?", (key_prefix + "%",))

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")


_default_cache: Optional[CacheDB] = None


def get_cache(db_path: str = "cache/autoresearch.db") -> CacheDB:
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheDB(db_path)
    return _default_cache
