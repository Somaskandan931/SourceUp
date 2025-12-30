"""
Session Management Module
--------------------------
Handles Redis/Memurai-based session storage.
Memurai is Redis-compatible and works on Windows.
"""

import redis
import json
import os

# Memurai runs on the same default port as Redis (6379)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# Initialize Redis/Memurai connection
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5
    )
    # Test connection
    r.ping()
    print(f"✅ Connected to Redis/Memurai at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"⚠️  Warning: Could not connect to Redis/Memurai: {e}")
    print("   Make sure Memurai service is running")
    r = None


def get_session(sid: str) -> dict:
    """
    Retrieve session data from Redis/Memurai.

    Args:
        sid: Session ID

    Returns:
        Session dictionary (empty dict if not found)
    """
    if r is None:
        print("⚠️  Redis/Memurai not available, returning empty session")
        return {}

    try:
        data = r.get(sid)
        return json.loads(data) if data else {}
    except Exception as e:
        print(f"Error retrieving session {sid}: {e}")
        return {}


def set_session(sid: str, data: dict):
    """
    Store session data in Redis/Memurai.

    Args:
        sid: Session ID
        data: Dictionary to store
    """
    if r is None:
        print("⚠️  Redis/Memurai not available, session not saved")
        return

    try:
        r.set(sid, json.dumps(data))
    except Exception as e:
        print(f"Error storing session {sid}: {e}")


def clear_session(sid: str):
    """
    Clear a specific session.

    Args:
        sid: Session ID
    """
    if r is None:
        return

    try:
        r.delete(sid)
    except Exception as e:
        print(f"Error clearing session {sid}: {e}")


def get_all_session_keys() -> list:
    """
    Get all session keys (for debugging).

    Returns:
        List of session keys
    """
    if r is None:
        return []

    try:
        return r.keys('*')
    except Exception as e:
        print(f"Error getting session keys: {e}")
        return []