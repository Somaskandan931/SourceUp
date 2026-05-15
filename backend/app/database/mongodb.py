"""
MongoDB Database Client for SourceUp-X
----------------------------------------
Async Motor client with:
  - Connection pooling
  - Index management (email unique, order_id unique)
  - Health check helper
  - Graceful reconnect support
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional


class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect(cls, uri: str, db_name: str):
        """
        Connect to MongoDB and create required indexes.

        Indexes created:
          - users.email      (unique) — enforces one account per email
          - orders.order_id  (unique) — prevents duplicate orders
          - orders.email     (non-unique) — fast lookup of user's orders
        """
        try:
            cls.client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )
            cls.db = cls.client[db_name]

            # Verify connection is alive
            await cls.client.admin.command("ping")
            print(f"✅ Connected to MongoDB: {db_name}")

            await cls._create_indexes()
            return cls.db

        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

    @classmethod
    async def _create_indexes(cls):
        """Create all required indexes if they don't already exist."""
        try:
            from pymongo import ASCENDING, IndexModel

            # ── users collection ────────────────────────────────────────
            user_indexes = [
                IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
            ]
            await cls.db.users.create_indexes(user_indexes)

            # ── orders collection ────────────────────────────────────────
            order_indexes = [
                IndexModel([("order_id", ASCENDING)], unique=True, name="order_id_unique"),
                IndexModel([("email", ASCENDING)], name="order_email_lookup"),
            ]
            await cls.db.orders.create_indexes(order_indexes)

            print("✅ MongoDB indexes created/verified")

        except Exception as e:
            # Indexes may already exist — this is not a fatal error
            print(f"⚠️ Index creation warning (may already exist): {e}")

    @classmethod
    async def close(cls):
        """Gracefully close the MongoDB connection."""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            print("✅ MongoDB connection closed")

    @classmethod
    def get_db(cls):
        """
        Return the database instance.
        Raises RuntimeError if not connected — call connect() first.
        """
        if cls.db is None:
            raise RuntimeError(
                "MongoDB is not connected. "
                "Call await MongoDB.connect(uri, db_name) at startup."
            )
        return cls.db

    @classmethod
    async def ping(cls) -> bool:
        """Return True if MongoDB is reachable, False otherwise."""
        try:
            if cls.client is None:
                return False
            await cls.client.admin.command("ping")
            return True
        except Exception:
            return False

    @classmethod
    async def get_collection(cls, name: str):
        """Convenience wrapper: get a named collection from the DB."""
        return cls.get_db()[name]