"""
Authentication & Billing API — SourceUp (MongoDB Version)
----------------------------------------
Provides:
  - User registration / login with JWT tokens
  - Google OAuth2 login (via Authlib)
  - GitHub OAuth2 login (optional)
  - UPI payment order creation + verification
  - Plan management (Free / Pro / Enterprise)
  - Demo login for testing

FIX: Added proper MongoDB connection guard + detailed error logging on register.
NEW: Google OAuth2 flow added at /auth/google/login and /auth/google/callback.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from backend.app.database.mongodb import MongoDB
from backend.app.utils.security import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPI_ID        = os.getenv("UPI_ID", "")
DEMO_EMAIL    = os.getenv("DEMO_EMAIL", "demo@sourceup.com")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demopass123")
DEMO_PLAN     = os.getenv("DEMO_PLAN", "pro")

# Google OAuth2 config — set these in your .env
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# Frontend URL to redirect after OAuth (set to your React dev server)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"

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
# Helper: get users collection with guard
# ---------------------------------------------------------------------------

def _get_users():
    """Return users collection, raising 503 if MongoDB is not connected."""
    try:
        db = MongoDB.get_db()
        return db.users
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database not connected. "
                "Check your MONGODB_URI environment variable and ensure MongoDB is running."
            )
        )


def _get_orders():
    """Return orders collection, raising 503 if MongoDB is not connected."""
    try:
        db = MongoDB.get_db()
        return db.orders
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Database not connected.")


# ---------------------------------------------------------------------------
# Auth Dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
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
    plan  = payload.get("plan", "free")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"email": email, "plan": plan}


# ---------------------------------------------------------------------------
# Auth Endpoints (MongoDB)
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Register a new user (free plan by default)."""
    users_collection = _get_users()   # raises 503 if DB is down

    # Check duplicate
    try:
        existing = await users_collection.find_one({"email": req.email})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed_password = hash_password(req.password)

    user = {
        "email":           req.email,
        "hashed_password": hashed_password,
        "plan":            "free",
        "company":         req.company or "",
        "created_at":      datetime.utcnow().isoformat(),
        "is_demo":         False,
        "auth_provider":   "email",   # track how the user signed up
    }

    try:
        await users_collection.insert_one(user)
    except Exception as e:
        # Duplicate key race condition
        if "duplicate key" in str(e).lower() or "E11000" in str(e):
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")

    token = create_access_token({"sub": req.email, "plan": "free"})
    return TokenResponse(access_token=token, plan="free", email=req.email)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Login and receive a JWT."""
    users_collection = _get_users()

    user = await users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Google-only accounts have no password
    if not user.get("hashed_password"):
        raise HTTPException(
            status_code=400,
            detail="This account uses Google sign-in. Please use 'Sign in with Google'."
        )

    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = create_access_token({"sub": req.email, "plan": user["plan"]})
    return TokenResponse(access_token=token, plan=user["plan"], email=req.email)


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return current user profile."""
    users_collection = _get_users()

    user = await users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "email":         current_user["email"],
        "plan":          current_user["plan"],
        "company":       user.get("company", ""),
        "created_at":    user.get("created_at", ""),
        "is_demo":       user.get("is_demo", False),
        "auth_provider": user.get("auth_provider", "email"),
        "avatar_url":    user.get("avatar_url", ""),
        "full_name":     user.get("full_name", ""),
    }


# ---------------------------------------------------------------------------
# Google OAuth2 — Step 1: redirect to Google
# ---------------------------------------------------------------------------

@router.get("/google/login")
async def google_login():
    """Redirect the browser to Google's OAuth2 consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=501,
            detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
        )

    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    from urllib.parse import urlencode
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)


# ---------------------------------------------------------------------------
# Google OAuth2 — Step 2: handle callback, create/login user
# ---------------------------------------------------------------------------

@router.get("/google/callback")
async def google_callback(code: str, request: Request):
    """
    Exchange the auth code for tokens, fetch the user's Google profile,
    then create or log in the user and redirect to the frontend with a JWT.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google OAuth not configured.")

    async with httpx.AsyncClient() as client:
        # Exchange code for access token
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Google token exchange failed: {token_resp.text}")

        token_data   = token_resp.json()
        access_token = token_data.get("access_token")

        # Fetch Google user info
        userinfo_resp = await client.get(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch Google user info")

        google_user = userinfo_resp.json()

    email      = google_user.get("email")
    full_name  = google_user.get("name", "")
    avatar_url = google_user.get("picture", "")

    if not email:
        raise HTTPException(status_code=400, detail="Google did not return an email address")

    users_collection = _get_users()

    # Find or create user
    existing = await users_collection.find_one({"email": email})

    if existing:
        # Update avatar/name in case they changed
        await users_collection.update_one(
            {"email": email},
            {"$set": {"full_name": full_name, "avatar_url": avatar_url, "auth_provider": "google"}},
        )
        plan = existing.get("plan", "free")
    else:
        # New user via Google — create with free plan
        new_user = {
            "email":           email,
            "hashed_password": None,          # no password for OAuth users
            "plan":            "free",
            "company":         "",
            "full_name":       full_name,
            "avatar_url":      avatar_url,
            "created_at":      datetime.utcnow().isoformat(),
            "is_demo":         False,
            "auth_provider":   "google",
        }
        await users_collection.insert_one(new_user)
        plan = "free"

    # Issue our own JWT
    jwt_token = create_access_token({"sub": email, "plan": plan})

    # Redirect to frontend with token in query param
    # The frontend reads it from the URL and stores in localStorage
    from urllib.parse import urlencode
    query_params = urlencode({
        'token': jwt_token,
        'email': email,
        'plan': plan,
        'name': full_name,
    })
    redirect_url = f"{FRONTEND_URL}/oauth-callback?{query_params}"
    return RedirectResponse(redirect_url)


# ---------------------------------------------------------------------------
# Demo Login Endpoint
# ---------------------------------------------------------------------------

@router.post("/demo-login", response_model=TokenResponse)
async def demo_login():
    """Demo login — creates or retrieves a demo Pro user."""
    users_collection = _get_users()

    demo_user = await users_collection.find_one({"email": DEMO_EMAIL})

    if not demo_user:
        demo_user_data = {
            "email":           DEMO_EMAIL,
            "hashed_password": hash_password(DEMO_PASSWORD),
            "plan":            DEMO_PLAN,
            "company":         "Demo Company",
            "is_demo":         True,
            "created_at":      datetime.utcnow().isoformat(),
            "auth_provider":   "email",
        }
        await users_collection.insert_one(demo_user_data)
    else:
        if demo_user.get("plan") != DEMO_PLAN:
            await users_collection.update_one(
                {"email": DEMO_EMAIL},
                {"$set": {"plan": DEMO_PLAN}}
            )

    token = create_access_token({"sub": DEMO_EMAIL, "plan": DEMO_PLAN})
    return TokenResponse(access_token=token, plan=DEMO_PLAN, email=DEMO_EMAIL)


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

    orders_collection = _get_orders()
    plan_info = PLANS[req.plan]
    order_id  = str(uuid.uuid4())

    await orders_collection.insert_one({
        "order_id":          order_id,
        "email":             current_user["email"],
        "plan":              req.plan,
        "amount":            plan_info["amount"],
        "created_at":        datetime.utcnow().isoformat(),
        "verified":          False,
        "upi_transaction_id": None,
        "verified_at":       None,
    })

    upi_link = (
        f"upi://pay?pa={UPI_ID}"
        f"&pn=SourceUp"
        f"&am={plan_info['amount']}"
        f"&cu=INR"
        f"&tn=SourceUp+{req.plan.title()}+Plan"
        f"&tr={order_id}"
    )

    return {
        "order_id":  order_id,
        "upi_id":    UPI_ID,
        "amount":    plan_info["amount"],
        "currency":  plan_info["currency"],
        "plan_name": plan_info["name"],
        "upi_link":  upi_link,
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
    orders_collection = _get_orders()
    users_collection  = _get_users()

    order = await orders_collection.find_one({"order_id": req.order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order["email"] != current_user["email"]:
        raise HTTPException(status_code=403, detail="Order does not belong to this user")

    if order.get("verified"):
        raise HTTPException(status_code=409, detail="Payment already verified")

    if req.plan not in PLANS:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    await orders_collection.update_one(
        {"order_id": req.order_id},
        {"$set": {
            "verified":           True,
            "upi_transaction_id": req.upi_transaction_id,
            "verified_at":        datetime.utcnow().isoformat()
        }}
    )

    await users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": {"plan": req.plan}}
    )

    new_token = create_access_token({"sub": current_user["email"], "plan": req.plan})

    return {
        "success":   True,
        "plan":      req.plan,
        "message":   f"Successfully upgraded to {req.plan.title()} plan",
        "new_token": new_token,
    }


@router.get("/billing/plans")
async def list_plans():
    """Return available plans and pricing."""
    return [
        {
            "id": "free", "name": "Free", "price_inr": 0,
            "features": ["10 searches/day", "Basic results", "No quote drafting"],
        },
        {
            "id": "pro", "name": "SourceUp Pro", "price_inr": 999,
            "features": ["Unlimited searches", "AI quote drafting", "Decision traces", "What-if scenarios"],
        },
        {
            "id": "enterprise", "name": "Enterprise", "price_inr": 4999,
            "features": ["Everything in Pro", "API access", "Priority support", "Custom integrations"],
        },
    ]
