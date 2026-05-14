"""
Session Management Module
--------------------------
Handles Redis/Memurai-based session storage with memory fallback.
"""

import redis
import json
import os
from typing import Dict, Any

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# In-memory fallback store
_memory_store: Dict[str, Any] = {}

# Initialize Redis/Memurai connection
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5
    )
    r.ping()
    print(f"✅ Connected to Redis/Memurai at {REDIS_HOST}:{REDIS_PORT}")
    USE_REDIS = True
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Redis/Memurai: {e}")
    print("   Using in-memory session storage fallback")
    USE_REDIS = False
    r = None


def get_session(sid: str) -> dict:
    """
    Retrieve session data from Redis or memory fallback.

    Args:
        sid: Session ID

    Returns:
        Session dictionary (empty dict if not found)
    """
    if USE_REDIS and r:
        try:
            data = r.get(f"session:{sid}")
            return json.loads(data) if data else {}
        except Exception as e:
            print(f"Error retrieving session from Redis {sid}: {e}")
            # Fall back to memory
            return _memory_store.get(sid, {})
    else:
        return _memory_store.get(sid, {})


def set_session(sid: str, data: dict, expiry_seconds: int = 3600):
    """
    Store session data in Redis or memory fallback.

    Args:
        sid: Session ID
        data: Dictionary to store
        expiry_seconds: Session expiry time (default 1 hour)
    """
    if USE_REDIS and r:
        try:
            r.setex(f"session:{sid}", expiry_seconds, json.dumps(data))
        except Exception as e:
            print(f"Error storing session in Redis {sid}: {e}")
            _memory_store[sid] = data
    else:
        _memory_store[sid] = data


def clear_session(sid: str):
    """Clear a specific session."""
    if USE_REDIS and r:
        try:
            r.delete(f"session:{sid}")
        except Exception as e:
            print(f"Error clearing session from Redis {sid}: {e}")

    if sid in _memory_store:
        del _memory_store[sid]


def get_all_session_keys() -> list:
    """Get all session keys (for debugging)."""
    if USE_REDIS and r:
        try:
            return [k.replace('session:', '') for k in r.keys('session:*')]
        except Exception as e:
            print(f"Error getting session keys from Redis: {e}")

    return list(_memory_store.keys())