"""
Authentication & Billing API — SourceUp (MongoDB Version)
----------------------------------------
Provides:
  - User registration / login with JWT tokens
  - UPI payment order creation + verification
  - Plan management (Free / Pro / Enterprise)
  - Demo login for testing
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from backend.app.database.mongodb import MongoDB
from backend.app.utils.security import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPI_ID = os.getenv("UPI_ID", "")
DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@sourceup.com")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demopass123")
DEMO_PLAN = os.getenv("DEMO_PLAN", "pro")

# Plan pricing in INR
PLANS = {
    "pro":        {"amount": 999,  "currency": "INR", "name": "SourceUp Pro"},
    "enterprise": {"amount": 4999, "currency": "INR", "name": "SourceUp Enterprise"},
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    company: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    plan: str
    email: str


class PaymentOrderRequest(BaseModel):
    plan: str


class PaymentVerifyRequest(BaseModel):
    order_id: str
    upi_transaction_id: str
    plan: str


# ---------------------------------------------------------------------------
# Auth Dependencies
# ---------------------------------------------------------------------------

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """Dependency: validate JWT and return user dict."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("sub")
    plan = payload.get("plan", "free")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"email": email, "plan": plan}


# ---------------------------------------------------------------------------
# Auth Endpoints (MongoDB)
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Register a new user (free plan by default)."""
    db = MongoDB.get_db()
    users_collection = db.users

    # Check if user exists
    existing = await users_collection.find_one({"email": req.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Hash password
    hashed_password = hash_password(req.password)

    # Create user document
    user = {
        "email": req.email,
        "hashed_password": hashed_password,
        "plan": "free",
        "company": req.company or "",
        "created_at": datetime.utcnow().isoformat(),
        "is_demo": False,
    }
    await users_collection.insert_one(user)

    # Create access token
    token = create_access_token({"sub": req.email, "plan": "free"})

    return TokenResponse(access_token=token, plan="free", email=req.email)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Login and receive a JWT."""
    db = MongoDB.get_db()
    users_collection = db.users

    # Find user
    user = await users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Create access token
    token = create_access_token({"sub": req.email, "plan": user["plan"]})

    return TokenResponse(access_token=token, plan=user["plan"], email=req.email)


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return current user profile."""
    db = MongoDB.get_db()
    users_collection = db.users

    user = await users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "email": current_user["email"],
        "plan": current_user["plan"],
        "company": user.get("company", ""),
        "created_at": user.get("created_at", ""),
        "is_demo": user.get("is_demo", False),
    }


# ---------------------------------------------------------------------------
# Demo Login Endpoint
# ---------------------------------------------------------------------------

@router.post("/demo-login", response_model=TokenResponse)
async def demo_login():
    """
    Demo login endpoint - creates or retrieves a demo Pro user.
    This is for demo purposes only - remove in production!
    """
    db = MongoDB.get_db()
    users_collection = db.users

    # Check if demo user exists
    demo_user = await users_collection.find_one({"email": DEMO_EMAIL})

    if not demo_user:
        # Create demo user with Pro plan
        demo_user_data = {
            "email": DEMO_EMAIL,
            "hashed_password": hash_password(DEMO_PASSWORD),
            "plan": DEMO_PLAN,
            "company": "Demo Company",
            "is_demo": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        await users_collection.insert_one(demo_user_data)
        print(f"✅ Created demo user: {DEMO_EMAIL} (plan: {DEMO_PLAN})")
    else:
        # Ensure demo user has Pro plan
        if demo_user.get("plan") != DEMO_PLAN:
            await users_collection.update_one(
                {"email": DEMO_EMAIL},
                {"$set": {"plan": DEMO_PLAN}}
            )
            print(f"✅ Updated demo user to {DEMO_PLAN} plan")

    token = create_access_token({"sub": DEMO_EMAIL, "plan": DEMO_PLAN})

    return TokenResponse(
        access_token=token,
        plan=DEMO_PLAN,
        email=DEMO_EMAIL
    )


# ---------------------------------------------------------------------------
# Billing Endpoints (MongoDB)
# ---------------------------------------------------------------------------

@router.post("/billing/order")
async def create_payment_order(
    req: PaymentOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a UPI payment order for plan upgrade."""
    if req.plan not in PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plan: {req.plan}. Choose: {list(PLANS.keys())}"
        )
    if not UPI_ID:
        raise HTTPException(
            status_code=503,
            detail="UPI_ID not configured. Add UPI_ID to your .env file."
        )

    db = MongoDB.get_db()
    orders_collection = db.orders

    plan_info = PLANS[req.plan]
    order_id = str(uuid.uuid4())

    # Store pending order
    await orders_collection.insert_one({
        "order_id": order_id,
        "email": current_user["email"],
        "plan": req.plan,
        "amount": plan_info["amount"],
        "created_at": datetime.utcnow().isoformat(),
        "verified": False,
        "upi_transaction_id": None,
        "verified_at": None,
    })

    # Build UPI deep-link
    upi_link = (
        f"upi://pay?pa={UPI_ID}"
        f"&pn=SourceUp"
        f"&am={plan_info['amount']}"
        f"&cu=INR"
        f"&tn=SourceUp+{req.plan.title()}+Plan"
        f"&tr={order_id}"
    )

    return {
        "order_id": order_id,
        "upi_id": UPI_ID,
        "amount": plan_info["amount"],
        "currency": plan_info["currency"],
        "plan_name": plan_info["name"],
        "upi_link": upi_link,
        "note": (
            f"Pay ₹{plan_info['amount']} to {UPI_ID} using any UPI app. "
            "Use the transaction ID from your UPI app to verify the payment below."
        ),
    }


@router.post("/billing/verify")
async def verify_payment(
    req: PaymentVerifyRequest,
    current_user: dict = Depends(get_current_user)
):
    """Verify UPI payment by order_id and UPI transaction ID (UTR)."""
    db = MongoDB.get_db()
    orders_collection = db.orders
    users_collection = db.users

    # Find order
    order = await orders_collection.find_one({"order_id": req.order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order["email"] != current_user["email"]:
        raise HTTPException(status_code=403, detail="Order does not belong to this user")

    if order.get("verified"):
        raise HTTPException(status_code=409, detail="Payment already verified")

    if req.plan not in PLANS:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    # Mark order as verified
    await orders_collection.update_one(
        {"order_id": req.order_id},
        {"$set": {
            "verified": True,
            "upi_transaction_id": req.upi_transaction_id,
            "verified_at": datetime.utcnow().isoformat()
        }}
    )

    # Upgrade user plan
    await users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": {"plan": req.plan}}
    )

    # Create new token with updated plan
    new_token = create_access_token({"sub": current_user["email"], "plan": req.plan})

    return {
        "success": True,
        "plan": req.plan,
        "message": f"Successfully upgraded to {req.plan.title()} plan",
        "new_token": new_token,
    }


@router.get("/billing/plans")
async def list_plans():
    """Return available plans and pricing."""
    return [
        {
            "id": "free",
            "name": "Free",
            "price_inr": 0,
            "features": ["10 searches/day", "Basic results", "No quote drafting"],
        },
        {
            "id": "pro",
            "name": "SourceUp Pro",
            "price_inr": 999,
            "features": [
                "Unlimited searches", "AI quote drafting",
                "Decision traces", "What-if scenarios"
            ],
        },
        {
            "id": "enterprise",
            "name": "Enterprise",
            "price_inr": 4999,
            "features": [
                "Everything in Pro", "API access",
                "Priority support", "Custom integrations"
            ],
        },
    ]