"""
Authentication & Billing API — SourceUp
----------------------------------------
Provides:
  - User registration / login with JWT tokens
  - UPI payment order creation + verification
  - Plan management (Free / Pro / Enterprise)

Dependencies:
    pip install python-jose[cryptography] passlib[bcrypt]

Environment variables required (add to .env):
    SECRET_KEY  — JWT signing secret (generate with: openssl rand -hex 32)
    UPI_ID      — your UPI VPA (e.g. yourname@upi or yourname@bank)
"""

import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# JWT (python-jose)
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("⚠️  python-jose not installed: pip install python-jose[cryptography]")

# Passlib (bcrypt)
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    print("⚠️  passlib not installed: pip install passlib[bcrypt]")

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY        = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM         = "HS256"
ACCESS_TOKEN_MINS = 60 * 24   # 24 hours

UPI_ID = os.getenv("UPI_ID", "")

# Plan pricing in INR
PLANS = {
    "pro":        {"amount": 999,  "currency": "INR", "name": "SourceUp Pro"},
    "enterprise": {"amount": 4999, "currency": "INR", "name": "SourceUp Enterprise"},
}

# ---------------------------------------------------------------------------
# In-memory user store (replace with DB in production)
# ---------------------------------------------------------------------------
_users: dict = {}         # email → {hashed_password, plan, created_at}
_pending_orders: dict = {}  # order_id → {email, plan, amount, created_at}


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
    plan: str   # "pro" or "enterprise"


class PaymentVerifyRequest(BaseModel):
    order_id: str
    upi_transaction_id: str   # UTR / transaction reference from UPI app
    plan: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    if PASSLIB_AVAILABLE:
        return pwd_context.hash(password)
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(plain: str, hashed: str) -> bool:
    if PASSLIB_AVAILABLE:
        return pwd_context.verify(plain, hashed)
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


def _create_token(email: str, plan: str) -> str:
    if not JWT_AVAILABLE:
        import base64
        return base64.b64encode(f"{email}:{plan}".encode()).decode()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_MINS)
    return jwt.encode(
        {"sub": email, "plan": plan, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )


def _decode_token(token: str) -> dict:
    if not JWT_AVAILABLE:
        import base64
        email, plan = base64.b64decode(token).decode().split(":", 1)
        return {"sub": email, "plan": plan}
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Dependency: validate JWT and return user dict."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        payload = _decode_token(credentials.credentials)
        email: str = payload.get("sub")
        plan:  str = payload.get("plan", "free")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email, "plan": plan}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    """Register a new user (free plan by default)."""
    if req.email in _users:
        raise HTTPException(status_code=409, detail="Email already registered")
    _users[req.email] = {
        "hashed_password": _hash_password(req.password),
        "plan":            "free",
        "company":         req.company or "",
        "created_at":      datetime.utcnow().isoformat(),
    }
    token = _create_token(req.email, "free")
    return TokenResponse(access_token=token, plan="free", email=req.email)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """Login and receive a JWT."""
    user = _users.get(req.email)
    if not user or not _verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _create_token(req.email, user["plan"])
    return TokenResponse(access_token=token, plan=user["plan"], email=req.email)


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    """Return current user profile."""
    email = current_user["email"]
    user_data = _users.get(email, {})
    return {
        "email":      email,
        "plan":       current_user["plan"],
        "company":    user_data.get("company", ""),
        "created_at": user_data.get("created_at", ""),
    }


# ---------------------------------------------------------------------------
# Billing endpoints — UPI
# ---------------------------------------------------------------------------

@router.post("/billing/order")
def create_payment_order(
    req: PaymentOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a UPI payment order for plan upgrade.
    Returns the UPI payment link / VPA and amount for the frontend
    to display a QR code or deep-link to a UPI app.
    """
    if req.plan not in PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plan: {req.plan}. Choose: {list(PLANS)}"
        )
    if not UPI_ID:
        raise HTTPException(
            status_code=503,
            detail="UPI_ID not configured. Add UPI_ID to your .env file."
        )

    plan_info = PLANS[req.plan]
    order_id  = str(uuid.uuid4())

    # Store pending order for later verification
    _pending_orders[order_id] = {
        "email":      current_user["email"],
        "plan":       req.plan,
        "amount":     plan_info["amount"],
        "created_at": datetime.utcnow().isoformat(),
        "verified":   False,
    }

    # Build UPI deep-link (works with any UPI app: GPay, PhonePe, Paytm, etc.)
    upi_link = (
        f"upi://pay?pa={UPI_ID}"
        f"&pn=SourceUp"
        f"&am={plan_info['amount']}"
        f"&cu=INR"
        f"&tn=SourceUp+{req.plan.title()}+Plan"
        f"&tr={order_id}"
    )

    return {
        "order_id":   order_id,
        "upi_id":     UPI_ID,
        "amount":     plan_info["amount"],
        "currency":   plan_info["currency"],
        "plan_name":  plan_info["name"],
        "upi_link":   upi_link,
        "note":       (
            f"Pay ₹{plan_info['amount']} to {UPI_ID} using any UPI app. "
            "Use the transaction ID from your UPI app to verify the payment below."
        ),
    }


@router.post("/billing/verify")
def verify_payment(
    req: PaymentVerifyRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Verify UPI payment by order_id and UPI transaction ID (UTR).
    The frontend submits the UTR after the user completes payment
    in their UPI app. Store and upgrade plan on receipt.

    NOTE: For production, integrate with your payment gateway's
    webhook or use a bank API to auto-verify UTRs.
    """
    order = _pending_orders.get(req.order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order["email"] != current_user["email"]:
        raise HTTPException(status_code=403, detail="Order does not belong to this user")
    if order["verified"]:
        raise HTTPException(status_code=409, detail="Payment already verified")
    if req.plan not in PLANS:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    # Mark order as verified and store transaction reference
    _pending_orders[req.order_id]["verified"]           = True
    _pending_orders[req.order_id]["upi_transaction_id"] = req.upi_transaction_id
    _pending_orders[req.order_id]["verified_at"]        = datetime.utcnow().isoformat()

    # Upgrade user plan
    email = current_user["email"]
    if email in _users:
        _users[email]["plan"] = req.plan

    new_token = _create_token(email, req.plan)

    return {
        "success":   True,
        "plan":      req.plan,
        "message":   f"Successfully upgraded to {req.plan.title()} plan",
        "new_token": new_token,
    }


@router.get("/billing/plans")
def list_plans():
    """Return available plans and pricing."""
    return [
        {
            "id":        "free",
            "name":      "Free",
            "price_inr": 0,
            "features":  ["10 searches/day", "Basic results", "No quote drafting"],
        },
        {
            "id":        "pro",
            "name":      "SourceUp Pro",
            "price_inr": 999,
            "features":  [
                "Unlimited searches", "AI quote drafting",
                "Decision traces", "What-if scenarios"
            ],
        },
        {
            "id":        "enterprise",
            "name":      "Enterprise",
            "price_inr": 4999,
            "features":  [
                "Everything in Pro", "API access",
                "Priority support", "Custom integrations"
            ],
        },
    ]
