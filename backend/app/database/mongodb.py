"""
MongoDB Database Client for SourceUp
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect(cls, uri: str, db_name: str):
        """Connect to MongoDB and create indexes"""
        try:
            cls.client = AsyncIOMotorClient(uri)
            cls.db = cls.client[db_name]
            print(f"✅ Connected to MongoDB: {db_name}")

            # Create indexes for performance
            try:
                await cls.db.users.create_index("email", unique=True)
                await cls.db.orders.create_index("order_id", unique=True)
                print("✅ MongoDB indexes created")
            except Exception as e:
                print(f"⚠️ Index creation warning: {e}")

            return cls.db
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            print("✅ MongoDB connection closed")

    @classmethod
    def get_db(cls):
        """Get database instance"""
        return cls.db