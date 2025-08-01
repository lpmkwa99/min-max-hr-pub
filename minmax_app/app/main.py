"""FastAPI implementation for the Min‑Max optimization tool.

This module exposes HTTP endpoints that correspond to the major
functionalities outlined in the project specifications, including
authentication, scenario management, simulation, calibration, and
recommendation/hot‑swap analysis. The implementation uses an in-memory
data store and is not intended for production use. To deploy a
production system, you should replace the data stores with a proper
database and implement secure authentication (for example, using JWTs
with private/public keys).
"""

from __future__ import annotations

import datetime as _dt
import secrets
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi import status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import sqlite3
import hashlib
import hmac
import base64



app = FastAPI(
    title="Gamified HR/Business Min‑Maxing Application API",
    description=(
        "This API demonstrates the core functionality of the Min‑Maxing "
        "platform as described in the provided specifications. It offers "
        "endpoints for user registration and login, scenario management, "
        "simulation of business outcomes based on slider inputs, data "
        "calibration using uploaded files, and a simple recommendation and "
        "hot‑swap analysis engine."
    ),
    version="0.1.0",
)

################################################################################
# Data Models and In‑Memory Store
################################################################################


class User(BaseModel):
    """Represents a user of the system."""

    id: int
    username: str
    password: str  # In a real system this would be hashed
    roles: List[str] = Field(default_factory=lambda: ["user"])
    tenant_id: int = 1  # default tenant for demonstration
    xp: int = 0
    level: int = 1


class Tenant(BaseModel):
    id: int
    name: str


class Scenario(BaseModel):
    id: int
    tenant_id: int
    user_id: int
    name: str
    sliders: Dict[str, float] = Field(default_factory=dict)
    created_at: _dt.datetime
    updated_at: _dt.datetime


class Calibration(BaseModel):
    id: int
    tenant_id: int
    filename: str
    content: bytes
    uploaded_at: _dt.datetime


class SimulationInput(BaseModel):
    """Defines the expected input payload for running a simulation."""

    sliders: Dict[str, float]
    scenario_name: Optional[str] = None


class SimulationResult(BaseModel):
    """Represents the output of a simulation run."""

    metrics: Dict[str, float]
    messages: List[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    """Represents a set of recommended solution providers/tools."""

    recommendations: List[str]
    rationale: str


class HotSwapResult(BaseModel):
    """Represents the result of a hot‑swap analysis."""

    missing_capabilities: List[str]
    performance_gaps: List[str]
    integration_issues: List[str]
    budget_misalignment: List[str]
    suggestions: List[str]


# Input schema for creating a new scenario version
class ScenarioVersionInput(BaseModel):
    """Schema for creating a new version of an existing scenario.

    ``sliders`` must include all slider values to persist for the version.
    A free‑form ``comment`` can be provided to describe the changes.
    """
    sliders: Dict[str, float]
    comment: Optional[str] = None

class LockRequest(BaseModel):
    """Schema for locking or unlocking a scenario version.

    ``locked`` indicates whether the version should be locked (True) or
    unlocked (False). Only users with admin or manager roles may lock
    or unlock versions. A locked version is intended to mark a
    snapshot as final and prevent accidental modifications. Creating
    new versions is still allowed; locking applies only to the
    selected version itself.
    """
    locked: bool

class RestoreRequest(BaseModel):
    """Schema for restoring a scenario to a specified version.

    ``version`` specifies which version's sliders should be copied back
    to the scenario's current state. The restore operation will
    overwrite the scenario's sliders and also create a new version
    capturing the restored state so that the action is auditable and
    reversible.
    """
    version: int


# In‑memory stores (deprecated)
#
# These dictionaries were used in the initial prototype to hold users,
# tenants, scenarios, and calibration data in memory. The refactored
# implementation now persists data in a SQLite database via the helper
# functions defined above (create_user, create_session, etc.). The
# dictionaries remain defined for backwards compatibility but are no
# longer used. They will be removed in a future refactor once all
# endpoints rely solely on the database.
users: Dict[int, User] = {}
tenants: Dict[int, Tenant] = {1: Tenant(id=1, name="Default Tenant")}
scenarios: Dict[int, Scenario] = {}
calibrations: Dict[int, Calibration] = {}
tokens: Dict[str, int] = {}

# ID counters (deprecated) – replaced by auto‑incrementing IDs in the
# database. They remain here only for compatibility with legacy logic
# and will be removed once the migration is complete.
_user_counter = 1
_scenario_counter = 1
_calibration_counter = 1


################################################################################
# Persistent Storage (SQLite)
################################################################################

# Path to the SQLite database file. It resides inside the package directory
DB_PATH = "./data.db"


def init_db() -> None:
    """Create required tables if they do not exist.

    This function initializes the SQLite database with tables for
    organizations, users, scenarios, sessions, and audit logs. It is
    idempotent and can be called at startup.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    # Organizations table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        """
    )
    # Users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            org_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            xp INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            FOREIGN KEY(org_id) REFERENCES organizations(id)
        );
        """
    )
    # Sessions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    # Scenarios table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            sliders TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(org_id) REFERENCES organizations(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    # Audit logs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            org_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(org_id) REFERENCES organizations(id)
        );
        """
    )
    # Scenario versions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scenario_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            sliders TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            comment TEXT,
            locked INTEGER DEFAULT 0,
            FOREIGN KEY(scenario_id) REFERENCES scenarios(id),
            FOREIGN KEY(created_by) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


    # Calibration parameters table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_params (
            org_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            value REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (org_id, param_name),
            FOREIGN KEY (org_id) REFERENCES organizations(id)
        );
        """
    )
    conn.commit()
    conn.close()


def get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row factory set to dict-like."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: str) -> str:
    """Return a SHA-256 hash of the password with provided salt."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username: str, password: str, org_name: str, role: str) -> int:
    """Create a new user with hashed password and return the new user ID.

    If the organization does not exist, it will be created. The password is
    salted and hashed before storage.
    """
    salt = secrets.token_hex(8)
    password_hash = hash_password(password, salt)
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure organization exists
    cur.execute("SELECT id FROM organizations WHERE name = ?", (org_name,))
    row = cur.fetchone()
    if row:
        org_id = row["id"]
    else:
        cur.execute("INSERT INTO organizations (name) VALUES (?)", (org_name,))
        org_id = cur.lastrowid
    # Insert user
    cur.execute(
        "INSERT INTO users (username, password_hash, salt, org_id, role) VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, salt, org_id, role),
    )
    user_id = cur.lastrowid
    conn.commit()
    conn.close()
    return user_id


def verify_user_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify a user's credentials and return user record if valid.

    Returns a dict with keys id, username, org_id, role if the password is
    correct. Otherwise returns None.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, salt, org_id, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    salt = row["salt"]
    expected_hash = row["password_hash"]
    if hash_password(password, salt) == expected_hash:
        user_record = {
            "id": row["id"],
            "username": username,
            "org_id": row["org_id"],
            "role": row["role"],
        }
        conn.close()
        return user_record
    conn.close()
    return None


def create_session(user_id: int, expires_in_seconds: int = 3600) -> str:
    """Create a new session token for the user and store it in DB."""
    token = secrets.token_hex(24)
    expires_at = (_dt.datetime.utcnow() + _dt.timedelta(seconds=expires_in_seconds)).isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Retrieve user data from a session token, verifying expiration."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT u.id, u.username, u.org_id, u.role, s.expires_at FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    expires_at = row["expires_at"]
    if expires_at and _dt.datetime.fromisoformat(expires_at) < _dt.datetime.utcnow():
        # Session expired; delete
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    user_record = {
        "id": row["id"],
        "username": row["username"],
        "org_id": row["org_id"],
        "role": row["role"],
    }
    conn.close()
    return user_record


def log_audit_event(user_id: int, org_id: int, action: str, details: Optional[str] = None) -> None:
    """Insert an audit log entry into the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit_logs (timestamp, user_id, org_id, action, details) VALUES (?, ?, ?, ?, ?)",
        (_dt.datetime.utcnow().isoformat(), user_id, org_id, action, details),
    )
    conn.commit()
    conn.close()


def set_calibration_params(org_id: int, params: Dict[str, float]) -> None:
    """Insert or update calibration parameters for an organization.

    Each key in ``params`` corresponds to a model parameter or baseline metric.
    Existing values are overwritten. A timestamp is recorded for auditing.
    """
    now = _dt.datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    for name, value in params.items():
        cur.execute(
            "INSERT INTO calibration_params (org_id, param_name, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(org_id, param_name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (org_id, name, float(value), now),
        )
    conn.commit()
    conn.close()


def get_calibration_params(org_id: int) -> Dict[str, float]:
    """Return all calibration parameters for the given organization.

    The result is a mapping from parameter names to their calibrated values.
    If no parameters are stored, an empty dict is returned.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT param_name, value FROM calibration_params WHERE org_id = ?",
        (org_id,),
    )
    params: Dict[str, float] = {}
    for row in cur.fetchall():
        params[row["param_name"]] = float(row["value"])
    conn.close()
    return params


# Ensure the database schema is created on module import. This call
# executes the CREATE TABLE statements in init_db() so that all
# subsequent operations against the database can succeed. It is safe
# to call init_db() multiple times because the CREATE TABLE commands
# use IF NOT EXISTS clauses.
init_db()



################################################################################
# Authentication helpers
################################################################################

security = HTTPBearer(auto_error=False)


def _generate_token() -> str:
    """Generate a random token string."""
    return secrets.token_hex(24)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Retrieve the current authenticated user based on a bearer token.

    This function extracts the token from the Authorization header and
    looks up the corresponding user session in the database. If the
    token is missing, invalid, or expired, an HTTP 401 error is raised.
    It returns a dictionary with keys ``id``, ``username``, ``org_id``,
    and ``role``.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    user_record = get_user_by_token(token)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_record


################################################################################
# Module definitions
################################################################################

MODULE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Core modules from the theory & design document
    "People": {
        "description": "Investment in people initiatives including training, wellness, and benefits.",
        "sliders": ["training_budget", "benefits_budget", "wellness_programs"],
    },
    "Marketing": {
        "description": "Investment in marketing and lead generation initiatives.",
        "sliders": ["lead_generation", "social_media", "events"],
    },
    # New business domain modules
    "Legal & Compliance": {
        "description": "Investments that reduce regulatory risk and compliance issues.",
        "sliders": ["compliance_investment", "policy_automation", "audit_training"],
    },
    # Customer Support & CX module: simplified to two high‑level sliders per spec
    "Customer Support & CX": {
        "description": "Investments that increase customer satisfaction, reduce churn, and improve onboarding.",
        "sliders": ["support_investment", "onboarding_quality"],
    },
    # Finance & Accounting module: simplified to two sliders representing automation and cost optimization
    "Finance & Accounting": {
        "description": "Investments in financial process automation and cost optimization initiatives.",
        "sliders": ["financial_process_automation", "cost_optimization"],
    },
    # IT Infrastructure & Security module: two sliders for modernization and cybersecurity spend
    "IT Infrastructure & Security": {
        "description": "Investments in IT modernization and cybersecurity to improve uptime and reduce incidents.",
        "sliders": ["it_modernization_budget", "cybersecurity_spend"],
    },
    # Procurement & Supply Chain module: two sliders for inventory and supplier management
    "Procurement & Supply Chain": {
        "description": "Investments to optimize inventory turnover and supplier relationships.",
        "sliders": ["inventory_optimization", "supplier_management"],
    },
}


################################################################################
# API Endpoints
################################################################################


class UserRegisterRequest(BaseModel):
    """Schema for user registration.

    In addition to a username and password, users must specify the
    organization they belong to and the role they wish to assume.
    Acceptable roles are ``admin``, ``manager``, or ``viewer``. If the
    role is omitted it defaults to ``viewer``. The organization name
    will be created if it does not already exist in the database.
    """

    username: str
    password: str
    organization: str = Field(alias="organization", min_length=1)
    role: str = Field(default="viewer")


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest) -> Dict[str, Any]:
    """Register a new user and return a session token.

    This endpoint creates a user record in the persistent database
    with a salted and hashed password. If the specified organization
    does not exist it will be created. A session token is generated
    and returned along with the user ID. An audit entry is recorded
    noting the registration action.
    """
    # Validate role
    role = payload.role.lower()
    if role not in {"admin", "manager", "viewer"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
    # Attempt to create the user; if the username already exists this will raise an IntegrityError
    try:
        user_id = create_user(payload.username, payload.password, payload.organization, role)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    # Create a session for the new user
    token = create_session(user_id)
    # Retrieve org_id for audit log
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT org_id FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    org_id = row["org_id"] if row else None
    if org_id is not None:
        log_audit_event(user_id, org_id, "register", details=f"User {payload.username} registered with role {role}")
    return {"token": token, "user_id": user_id}


class UserLoginRequest(BaseModel):
    """Schema for user login."""

    username: str
    password: str


@app.post("/login")
def login(payload: UserLoginRequest) -> Dict[str, str]:
    """Authenticate a user and return a session token.

    The provided username and password are validated against the
    database. If valid, a new session record is created with an
    expiration time. An audit entry is added for the login action.
    """
    user_record = verify_user_credentials(payload.username, payload.password)
    if not user_record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    user_id = user_record["id"]
    org_id = user_record["org_id"]
    token = create_session(user_id)
    # Log the login event
    log_audit_event(user_id, org_id, "login", details=f"User {payload.username} logged in")
    return {"token": token}


@app.get("/me")
def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the profile of the currently authenticated user.

    The returned object includes the user ID, username, organization
    identifier, and role. Additional fields such as XP and level are
    omitted for brevity but may be added later.
    """
    return current_user


@app.get("/modules")
def list_modules() -> Dict[str, Any]:
    """List all available modules and their slider definitions."""
    return MODULE_DEFINITIONS


@app.post("/scenario", status_code=status.HTTP_201_CREATED)
def create_scenario(
    payload: SimulationInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new scenario for the current user.

    The scenario is persisted in the database associated with the
    caller's organization and user ID. Slider values are stored as
    JSON. A corresponding audit entry is recorded.
    """
    import json
    now = _dt.datetime.utcnow().isoformat()
    name = payload.scenario_name or "Untitled Scenario"
    sliders_json = json.dumps(payload.sliders)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scenarios (org_id, user_id, name, sliders, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            current_user["org_id"],
            current_user["id"],
            name,
            sliders_json,
            now,
            now,
        ),
    )
    scenario_id = cur.lastrowid
    # Also insert initial version (version 1)
    cur.execute(
        "INSERT INTO scenario_versions (scenario_id, version, sliders, created_at, created_by, comment) VALUES (?, ?, ?, ?, ?, ?)",
        (
            scenario_id,
            1,
            sliders_json,
            now,
            current_user["id"],
            "Initial version",
        ),
    )
    conn.commit()
    conn.close()
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "scenario_create", details=f"Created scenario {name} (ID {scenario_id})")
    return {
        "id": scenario_id,
        "tenant_id": current_user["org_id"],
        "user_id": current_user["id"],
        "name": name,
        "sliders": payload.sliders,
        "created_at": now,
        "updated_at": now,
    }


@app.get("/scenario/{scenario_id}")
def get_scenario(scenario_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Retrieve a scenario by its ID.

    Users may only access scenarios within their own organization. The
    scenario is returned as a dictionary with its sliders parsed
    back into a Python object.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, org_id, user_id, name, sliders, created_at, updated_at FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    import json
    sliders = json.loads(row["sliders"]) if row["sliders"] else {}
    return {
        "id": row["id"],
        "tenant_id": row["org_id"],
        "user_id": row["user_id"],
        "name": row["name"],
        "sliders": sliders,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.post("/scenario/{scenario_id}/clone", status_code=status.HTTP_201_CREATED)
def clone_scenario(scenario_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Clone an existing scenario and return the new scenario.

    The cloned scenario copies slider settings and updates the name. The
    user performing the clone becomes the owner of the new scenario.
    An audit entry is recorded.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Fetch original scenario
    cur.execute("SELECT id, org_id, user_id, name, sliders FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    import json
    original_name = row["name"]
    sliders_json = row["sliders"]
    now = _dt.datetime.utcnow().isoformat()
    # Insert cloned scenario; name will be suffixed with clone id once inserted
    # We first insert to get the new ID
    clone_base_name = f"{original_name} (Clone)"
    cur.execute(
        "INSERT INTO scenarios (org_id, user_id, name, sliders, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            current_user["org_id"],
            current_user["id"],
            clone_base_name,
            sliders_json,
            now,
            now,
        ),
    )
    new_id = cur.lastrowid
    # Update name to include unique clone ID
    clone_name = f"{original_name} (Clone {new_id})"
    cur.execute("UPDATE scenarios SET name = ? WHERE id = ?", (clone_name, new_id))
    conn.commit()
    conn.close()
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "scenario_clone", details=f"Cloned scenario {scenario_id} to {new_id}")
    sliders = json.loads(sliders_json) if sliders_json else {}
    return {
        "id": new_id,
        "tenant_id": current_user["org_id"],
        "user_id": current_user["id"],
        "name": clone_name,
        "sliders": sliders,
        "created_at": now,
        "updated_at": now,
    }


@app.post("/simulate", response_model=SimulationResult)
def run_simulation(
    payload: SimulationInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> SimulationResult:
    """Run a simulation based on provided slider values.

    The algorithm implemented here is intentionally simplistic. It
    illustrates how slider values might be translated into business
    metrics with linear relationships and diminishing returns. In a
    production setting this function would call into a proper
    simulation engine calibrated with organizational data.
    """
    sliders = payload.sliders
    # Import math for diminishing return calculations
    import math

    # Initialize metrics with sensible baseline values. Some metrics are on
    # 0–100 scales (e.g. satisfaction, ROI), while others use domain‑specific
    # units (e.g. hours or days) and are later converted to a normalized score.
    base_values = {
        "employee_satisfaction": 50.0,
        "customer_satisfaction": 50.0,
        "nps": -10.0,
        "churn_rate": 15.0,  # percentage of customers leaving per period
        "first_response_time": 24.0,  # hours to first response
        "net_profit_margin": 10.0,  # percentage
        "cogs_reduction": 0.0,  # percentage reduction in COGS
        "budget_variance": 20.0,  # percentage variance (lower is better)
        "system_uptime": 95.0,  # uptime percentage
        "security_incidents": 5.0,  # incidents per period
        "productivity_boost": 0.0,  # percentage increase
        "inventory_turnover_ratio": 5.0,  # times per year
        "procurement_cost_reduction": 0.0,  # percentage reduction
        "lead_time": 10.0,  # days
        "roi": 0.0,
    }

    # Apply calibration baseline overrides if present for this org
    calibrations = get_calibration_params(current_user["org_id"])
    for param_name, value in calibrations.items():
        if param_name.startswith("baseline_"):
            metric = param_name[len("baseline_"):]
            if metric in base_values:
                base_values[metric] = float(value)

    # Helper functions for diminishing returns
    def saturating_effect(value: float, alpha: float = 0.05) -> float:
        """Return a value in [0,1] exhibiting diminishing returns.

        Uses an exponential saturation curve: effect = 1 - exp(-alpha * value).
        """
        return 1.0 - math.exp(-alpha * value)

    def optimal_effect(value: float) -> float:
        """Return a value in [0,1] that peaks at 50% and falls off toward 0 at extremes.

        Uses a sine curve to model a sweet spot: sin(pi * value/100).
        """
        return math.sin(math.pi * value / 100.0)

    # Extract and normalize slider values (0–100)
    def get_slider(name: str) -> float:
        return float(sliders.get(name, 0.0))

    # 1. People module (training, benefits, wellness)
    training = get_slider("training_budget")
    benefits = get_slider("benefits_budget")
    wellness = get_slider("wellness_programs")
    people_level = (training + benefits + wellness) / 3.0
    people_effect = saturating_effect(people_level)
    base_values["employee_satisfaction"] += 30.0 * people_effect
    base_values["roi"] += 5.0 * people_effect

    # 2. Marketing module (lead generation, social media, events)
    lead_gen = get_slider("lead_generation")
    social_media = get_slider("social_media")
    events = get_slider("events")
    marketing_level = (lead_gen * 0.6 + social_media * 0.3 + events * 0.1)
    marketing_effect = saturating_effect(marketing_level)
    base_values["roi"] += 50.0 * marketing_effect

    # 3. Customer Support & CX module
    support_inv = get_slider("support_investment")
    onboarding_quality = get_slider("onboarding_quality")
    support_effect = saturating_effect(support_inv)
    onboarding_effect = saturating_effect(onboarding_quality)
    # Customer satisfaction (CSAT) out of 100
    base_values["customer_satisfaction"] += 40.0 * support_effect + 20.0 * onboarding_effect
    base_values["customer_satisfaction"] = min(base_values["customer_satisfaction"], 100.0)
    # NPS ranges from -100 to +100
    base_values["nps"] += 60.0 * support_effect + 40.0 * onboarding_effect
    base_values["nps"] = max(min(base_values["nps"], 100.0), -100.0)
    # Churn rate decreases with better support/onboarding (cannot be <0)
    churn_reduction = 0.8 * support_effect + 0.2 * onboarding_effect
    base_values["churn_rate"] = max(0.0, base_values["churn_rate"] * (1.0 - churn_reduction))
    # First response time (hours) decreases with support investment
    response_reduction = 0.8 * support_effect
    base_values["first_response_time"] = base_values["first_response_time"] * (1.0 - response_reduction)
    # ROI modest boost from happier customers
    base_values["roi"] += 3.0 * support_effect

    # 4. Finance & Accounting module
    automation_level = get_slider("financial_process_automation")
    cost_opt_level = get_slider("cost_optimization")
    automation_effect = saturating_effect(automation_level)
    cost_opt_effect = saturating_effect(cost_opt_level)
    # Net profit margin improves with automation and cost optimization
    base_values["net_profit_margin"] += 5.0 * automation_effect + 10.0 * cost_opt_effect
    # COGS reduction grows with cost optimization (max ~20%)
    base_values["cogs_reduction"] += 20.0 * cost_opt_effect
    # Budget variance decreases with automation (better planning) – lower is better
    base_values["budget_variance"] = base_values["budget_variance"] * (1.0 - 0.8 * automation_effect)
    # ROI increases slightly due to improved margins
    base_values["roi"] += 4.0 * (automation_effect + cost_opt_effect)

    # 5. IT Infrastructure & Security module
    modernization_level = get_slider("it_modernization_budget")
    cyber_level = get_slider("cybersecurity_spend")
    modern_effect = saturating_effect(modernization_level)
    cyber_effect = saturating_effect(cyber_level)
    # System uptime increases with modernization, slightly decreases with heavy security overhead
    base_values["system_uptime"] += 5.0 * modern_effect - 2.0 * cyber_effect
    base_values["system_uptime"] = min(max(base_values["system_uptime"], 0.0), 100.0)
    # Security incidents decrease with cybersecurity spend
    base_values["security_incidents"] = base_values["security_incidents"] * (1.0 - 0.8 * cyber_effect)
    # Productivity boost from modernization
    base_values["productivity_boost"] += 20.0 * modern_effect
    # ROI sees a small uplift from productivity gains
    base_values["roi"] += 2.0 * modern_effect

    # 6. Procurement & Supply Chain module
    inventory_level = get_slider("inventory_optimization")
    supplier_level = get_slider("supplier_management")
    # Inventory turnover ratio has an optimal sweet spot at 50
    turnover_effect = optimal_effect(inventory_level)
    base_values["inventory_turnover_ratio"] += 4.0 * turnover_effect  # adds up to 4 times
    # Procurement cost reduction from supplier management (max ~20%)
    supplier_effect = saturating_effect(supplier_level)
    base_values["procurement_cost_reduction"] += 20.0 * supplier_effect
    # Lead time decreases as supplier management improves (cannot go <2 days)
    base_values["lead_time"] = max(2.0, base_values["lead_time"] * (1.0 - 0.6 * supplier_effect))
    # ROI benefits modestly from cost savings
    base_values["roi"] += 3.0 * (turnover_effect + supplier_effect)

    # Aggregate ROI adjustments from all modules (clamp between 0–100)
    base_values["roi"] = max(min(base_values["roi"], 100.0), 0.0)

    # Post‑processing: convert certain metrics to normalized 0–100 scales for dashboard
    metrics = {}
    # Employee satisfaction 0–100
    metrics["employee_satisfaction"] = max(min(base_values["employee_satisfaction"], 100.0), 0.0)
    # Customer satisfaction (CSAT) 0–100
    metrics["customer_satisfaction"] = max(min(base_values["customer_satisfaction"], 100.0), 0.0)
    # Net Promoter Score retains its -100 to 100 range but we include it as is
    metrics["nps"] = base_values["nps"]
    # Churn rate as a percentage (0–100) – lower is better
    metrics["churn_rate"] = max(min(base_values["churn_rate"], 100.0), 0.0)
    # First response time scaled to score (0 best, 24 worst -> 0–100 by inverse)
    metrics["first_response_score"] = max(min(100.0 - base_values["first_response_time"] * 4.0, 100.0), 0.0)
    # Net profit margin (can be negative up to -50)
    metrics["net_profit_margin"] = max(min(base_values["net_profit_margin"], 100.0), -50.0)
    # COGS reduction percentage
    metrics["cogs_reduction"] = max(min(base_values["cogs_reduction"], 100.0), 0.0)
    # Budget adherence score (higher is better) = 100 - variance
    metrics["budget_adherence"] = max(min(100.0 - base_values["budget_variance"], 100.0), 0.0)
    # System uptime percentage
    metrics["system_uptime"] = max(min(base_values["system_uptime"], 100.0), 0.0)
    # Security score: scale incidents to a 0–100 score (0 incidents -> 100)
    metrics["security_score"] = max(min(100.0 - base_values["security_incidents"] * 20.0, 100.0), 0.0)
    # Productivity boost percentage
    metrics["productivity_boost"] = max(min(base_values["productivity_boost"], 100.0), 0.0)
    # Inventory turnover score (ratio *10 to approximate 0–100)
    metrics["inventory_turnover"] = max(min(base_values["inventory_turnover_ratio"] * 10.0, 100.0), 0.0)
    # Procurement cost reduction percentage
    metrics["procurement_cost_reduction"] = max(min(base_values["procurement_cost_reduction"], 100.0), 0.0)
    # Lead time score (inverse of days, scaled to 0–100)
    metrics["lead_time_score"] = max(min(100.0 - base_values["lead_time"] * 10.0, 100.0), 0.0)
    # ROI percentage (0–100)
    metrics["roi"] = base_values["roi"]

    # Final messages list (placeholder for warnings or notes)
    messages: List[str] = []
    # Audit log – include scenario_name if provided
    scenario_name = payload.scenario_name or "ad‑hoc"
    log_audit_event(current_user["id"], current_user["org_id"], "simulate", details=f"Ran simulation ({scenario_name})")
    return SimulationResult(metrics=metrics, messages=messages)


# Internal helper to compute simulation metrics without user context.
def compute_simulation_metrics(sliders: Dict[str, float], org_id: Optional[int] = None) -> Dict[str, float]:
    """Compute business metrics from slider values with optional calibration.

    This helper mirrors the logic used in the /simulate endpoint but returns only
    the metrics dict. It accepts an optional ``org_id``; if provided, any
    calibration parameters for that organization will override baseline values
    before calculations. It is used by comparison and reporting functions to
    evaluate scenario versions without repeating API calls.
    """
    import math
    # Initialize baseline values
    base_values = {
        "employee_satisfaction": 50.0,
        "customer_satisfaction": 50.0,
        "nps": -10.0,
        "churn_rate": 15.0,
        "first_response_time": 24.0,
        "net_profit_margin": 10.0,
        "cogs_reduction": 0.0,
        "budget_variance": 20.0,
        "system_uptime": 95.0,
        "security_incidents": 5.0,
        "productivity_boost": 0.0,
        "inventory_turnover_ratio": 5.0,
        "procurement_cost_reduction": 0.0,
        "lead_time": 10.0,
        "roi": 0.0,
    }
    # Apply calibration baseline overrides
    if org_id is not None:
        calibrations = get_calibration_params(org_id)
        for param_name, value in calibrations.items():
            if param_name.startswith("baseline_"):
                metric = param_name[len("baseline_"):]
                if metric in base_values:
                    base_values[metric] = float(value)
    def saturating_effect(value: float, alpha: float = 0.05) -> float:
        return 1.0 - math.exp(-alpha * value)
    def optimal_effect(value: float) -> float:
        return math.sin(math.pi * value / 100.0)
    def get_slider(name: str) -> float:
        return float(sliders.get(name, 0.0))
    # People module
    training = get_slider("training_budget")
    benefits = get_slider("benefits_budget")
    wellness = get_slider("wellness_programs")
    people_level = (training + benefits + wellness) / 3.0
    people_effect = saturating_effect(people_level)
    base_values["employee_satisfaction"] += 30.0 * people_effect
    base_values["roi"] += 5.0 * people_effect
    # Marketing module
    lead_gen = get_slider("lead_generation")
    social_media = get_slider("social_media")
    events = get_slider("events")
    marketing_level = (lead_gen * 0.6 + social_media * 0.3 + events * 0.1)
    marketing_effect = saturating_effect(marketing_level)
    base_values["roi"] += 50.0 * marketing_effect
    # Customer Support & CX module
    support_inv = get_slider("support_investment")
    onboarding_quality = get_slider("onboarding_quality")
    support_effect = saturating_effect(support_inv)
    onboarding_effect = saturating_effect(onboarding_quality)
    base_values["customer_satisfaction"] += 40.0 * support_effect + 20.0 * onboarding_effect
    base_values["customer_satisfaction"] = min(base_values["customer_satisfaction"], 100.0)
    base_values["nps"] += 60.0 * support_effect + 40.0 * onboarding_effect
    base_values["nps"] = max(min(base_values["nps"], 100.0), -100.0)
    churn_reduction = 0.8 * support_effect + 0.2 * onboarding_effect
    base_values["churn_rate"] = max(0.0, base_values["churn_rate"] * (1.0 - churn_reduction))
    response_reduction = 0.8 * support_effect
    base_values["first_response_time"] = base_values["first_response_time"] * (1.0 - response_reduction)
    base_values["roi"] += 3.0 * support_effect
    # Finance & Accounting module
    automation_level = get_slider("financial_process_automation")
    cost_opt_level = get_slider("cost_optimization")
    automation_effect = saturating_effect(automation_level)
    cost_opt_effect = saturating_effect(cost_opt_level)
    base_values["net_profit_margin"] += 5.0 * automation_effect + 10.0 * cost_opt_effect
    base_values["cogs_reduction"] += 20.0 * cost_opt_effect
    base_values["budget_variance"] = base_values["budget_variance"] * (1.0 - 0.8 * automation_effect)
    base_values["roi"] += 4.0 * (automation_effect + cost_opt_effect)
    # IT Infrastructure & Security module
    modernization_level = get_slider("it_modernization_budget")
    cyber_level = get_slider("cybersecurity_spend")
    modern_effect = saturating_effect(modernization_level)
    cyber_effect = saturating_effect(cyber_level)
    base_values["system_uptime"] += 5.0 * modern_effect - 2.0 * cyber_effect
    base_values["system_uptime"] = min(max(base_values["system_uptime"], 0.0), 100.0)
    base_values["security_incidents"] = base_values["security_incidents"] * (1.0 - 0.8 * cyber_effect)
    base_values["productivity_boost"] += 20.0 * modern_effect
    base_values["roi"] += 2.0 * modern_effect
    # Procurement & Supply Chain module
    inventory_level = get_slider("inventory_optimization")
    supplier_level = get_slider("supplier_management")
    turnover_effect = optimal_effect(inventory_level)
    base_values["inventory_turnover_ratio"] += 4.0 * turnover_effect
    supplier_effect = saturating_effect(supplier_level)
    base_values["procurement_cost_reduction"] += 20.0 * supplier_effect
    base_values["lead_time"] = max(2.0, base_values["lead_time"] * (1.0 - 0.6 * supplier_effect))
    base_values["roi"] += 3.0 * (turnover_effect + supplier_effect)
    # Clamp ROI to 0–100
    base_values["roi"] = max(min(base_values["roi"], 100.0), 0.0)
    # Post-processing to derive metrics
    metrics: Dict[str, float] = {}
    metrics["employee_satisfaction"] = max(min(base_values["employee_satisfaction"], 100.0), 0.0)
    metrics["customer_satisfaction"] = max(min(base_values["customer_satisfaction"], 100.0), 0.0)
    metrics["nps"] = base_values["nps"]
    metrics["churn_rate"] = max(min(base_values["churn_rate"], 100.0), 0.0)
    metrics["first_response_score"] = max(min(100.0 - base_values["first_response_time"] * 4.0, 100.0), 0.0)
    metrics["net_profit_margin"] = max(min(base_values["net_profit_margin"], 100.0), -50.0)
    metrics["cogs_reduction"] = max(min(base_values["cogs_reduction"], 100.0), 0.0)
    metrics["budget_adherence"] = max(min(100.0 - base_values["budget_variance"], 100.0), 0.0)
    metrics["system_uptime"] = max(min(base_values["system_uptime"], 100.0), 0.0)
    metrics["security_score"] = max(min(100.0 - base_values["security_incidents"] * 20.0, 100.0), 0.0)
    metrics["productivity_boost"] = max(min(base_values["productivity_boost"], 100.0), 0.0)
    metrics["inventory_turnover"] = max(min(base_values["inventory_turnover_ratio"] * 10.0, 100.0), 0.0)
    metrics["procurement_cost_reduction"] = max(min(base_values["procurement_cost_reduction"], 100.0), 0.0)
    metrics["lead_time_score"] = max(min(100.0 - base_values["lead_time"] * 10.0, 100.0), 0.0)
    metrics["roi"] = base_values["roi"]
    return metrics


class CalibrationInput(BaseModel):
    """Schema for uploading calibration data.

    ``content`` should be a base64‑encoded string representing the
    contents of the calibration file. ``filename`` is used for
    identification only and is not parsed on upload.
    """

    filename: Optional[str] = None
    content: Optional[str] = None
    parameters: Optional[Dict[str, float]] = None


@app.post("/calibrate", status_code=status.HTTP_201_CREATED)
def upload_calibration(
    payload: CalibrationInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Upload or set calibration data for the current user's organization.

    Calibration data can be provided in one of two formats:

    1. A ``parameters`` dictionary mapping model parameter names to values. This
       allows clients to directly specify baseline metrics or multipliers
       without uploading a file.
    2. A ``content`` string containing base64‑encoded CSV data. The CSV
       should have two columns: ``param_name`` and ``value`` (no header is
       required). Each line is parsed into a parameter entry. Invalid lines
       are ignored. A filename may be provided for reference.

    After parsing, all parameter values are stored in the calibration table
    for the user's organization using ``set_calibration_params``. Previous
    values for the same parameters are overwritten. An audit event is
    recorded. The response includes the stored parameter names and values.
    """
    import base64
    import csv
    params: Dict[str, float] = {}
    # Case 1: direct parameters
    if payload.parameters:
        for key, value in payload.parameters.items():
            try:
                params[key] = float(value)
            except Exception:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid parameter value for {key}: {value}")
    # Case 2: base64 CSV content
    elif payload.content:
        try:
            decoded = base64.b64decode(payload.content.encode("utf-8"), validate=True).decode("utf-8")
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid base64 content: {exc}")
        # Parse CSV: expect "param_name,value" on each line
        reader = csv.reader(decoded.splitlines())
        for row in reader:
            if not row or len(row) < 2:
                continue
            name = row[0].strip()
            try:
                value = float(row[1].strip())
            except Exception:
                continue
            params[name] = value
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No calibration data provided")
    # Persist calibration parameters
    set_calibration_params(current_user["org_id"], params)
    # Audit log
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "calibration_update",
        details=f"Updated calibration parameters: {', '.join(params.keys())}",
    )
    return {
        "organization_id": current_user["org_id"],
        "updated_parameters": params,
    }


@app.get("/recommendations", response_model=RecommendationResult)
def get_recommendations(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RecommendationResult:
    """Return recommended solution providers/tools.

    This stubbed implementation returns a static list of vendors. In a
    complete system the recommendations would be generated based on the
    user's scenario history, calibration data, and current market data.
    """
    recs = [
        "Acme HR Platform",
        "NextGen Analytics Suite",
        "SecureCloud CRM",
        "LeanProcure Procurement",
    ]
    rationale = "These providers align with your current investment profile and offer strong ROI across multiple metrics."
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "recommendations", details="Viewed recommendations")
    return RecommendationResult(recommendations=recs, rationale=rationale)


# Calibration retrieval endpoint
@app.get("/calibration")
def fetch_calibration(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all calibration parameters for the current user's organization.

    This endpoint returns a mapping of parameter names to their calibrated
    values. If no parameters have been set, an empty object is returned.
    """
    params = get_calibration_params(current_user["org_id"])
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "calibration_fetch", details="Fetched calibration parameters")
    return params


@app.get("/hot_swap", response_model=HotSwapResult)
def hot_swap_analysis(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> HotSwapResult:
    """Perform a simple hot‑swap gap analysis for the current tenant.

    This function identifies generic gaps in the user's current tool
    stack and returns suggestions. The implementation does not yet
    inspect real user data; it simply returns a static analysis
    representing missing capabilities and integration challenges.
    """
    missing = ["Advanced analytics", "HR automation"]
    gaps = ["Low lead conversion rate", "High employee churn"]
    issues = ["CRM does not integrate with ERP", "Manual data entry causes delays"]
    budget = ["Marketing budget underutilized"]
    suggestions = [
        "Adopt NextGen Analytics Suite to improve analytics and lead conversion.",
        "Replace manual HR processes with Acme HR Platform to reduce churn.",
        "Integrate CRM with ERP using SecureCloud connectors.",
    ]
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "hot_swap_analysis", details="Performed hot swap analysis")
    return HotSwapResult(
        missing_capabilities=missing,
        performance_gaps=gaps,
        integration_issues=issues,
        budget_misalignment=budget,
        suggestions=suggestions,
    )


################################################################################
# Scenario Versioning Endpoints
################################################################################


@app.post("/scenario/{scenario_id}/versions", status_code=status.HTTP_201_CREATED)
def create_scenario_version(
    scenario_id: int,
    payload: ScenarioVersionInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new version for an existing scenario.

    The request must include a complete set of sliders. A version number is
    assigned automatically based on the highest existing version for the
    scenario. Only users within the same organization as the scenario may
    create versions. A comment describing the changes may be provided.
    """
    import json
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario exists and belongs to org
    cur.execute("SELECT org_id FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Determine next version number
    cur.execute("SELECT MAX(version) as max_version FROM scenario_versions WHERE scenario_id = ?", (scenario_id,))
    vrow = cur.fetchone()
    next_version = (vrow["max_version"] or 0) + 1
    now = _dt.datetime.utcnow().isoformat()
    sliders_json = json.dumps(payload.sliders)
    cur.execute(
        "INSERT INTO scenario_versions (scenario_id, version, sliders, created_at, created_by, comment) VALUES (?, ?, ?, ?, ?, ?)",
        (
            scenario_id,
            next_version,
            sliders_json,
            now,
            current_user["id"],
            payload.comment,
        ),
    )
    version_id = cur.lastrowid
    conn.commit()
    conn.close()
    # Audit log
    log_audit_event(current_user["id"], current_user["org_id"], "scenario_version_create", details=f"Scenario {scenario_id} version {next_version} created")
    return {
        "id": version_id,
        "scenario_id": scenario_id,
        "version": next_version,
        "sliders": payload.sliders,
        "created_at": now,
        "created_by": current_user["id"],
        "comment": payload.comment,
    }


@app.get("/scenario/{scenario_id}/versions")
def list_scenario_versions(
    scenario_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all versions for a given scenario.

    Returns version metadata (version number, creation date, creator, comment) in
    ascending order. Only accessible by users within the scenario's
    organization.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario and org
    cur.execute("SELECT org_id FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Fetch versions
    cur.execute(
        "SELECT id, version, created_at, created_by, comment FROM scenario_versions WHERE scenario_id = ? ORDER BY version ASC",
        (scenario_id,),
    )
    versions = []
    for vrow in cur.fetchall():
        versions.append(
            {
                "id": vrow["id"],
                "scenario_id": scenario_id,
                "version": vrow["version"],
                "created_at": vrow["created_at"],
                "created_by": vrow["created_by"],
                "comment": vrow["comment"],
            }
        )
    conn.close()
    return versions


@app.get("/scenario/{scenario_id}/versions/{version_number}")
def get_scenario_version(
    scenario_id: int,
    version_number: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve a specific version of a scenario.

    Returns the slider values and metadata for the requested version. Users
    outside of the scenario's organization cannot access the data.
    """
    import json
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario and org
    cur.execute("SELECT org_id FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Fetch version
    cur.execute(
        "SELECT id, version, sliders, created_at, created_by, comment FROM scenario_versions WHERE scenario_id = ? AND version = ?",
        (scenario_id, version_number),
    )
    vrow = cur.fetchone()
    conn.close()
    if not vrow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    sliders = json.loads(vrow["sliders"]) if vrow["sliders"] else {}
    return {
        "id": vrow["id"],
        "scenario_id": scenario_id,
        "version": vrow["version"],
        "sliders": sliders,
        "created_at": vrow["created_at"],
        "created_by": vrow["created_by"],
        "comment": vrow["comment"],
    }


@app.get("/scenario/{scenario_id}/compare")
def compare_scenario_versions(
    scenario_id: int,
    version_a: int,
    version_b: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Compare two versions of the same scenario.

    The comparison returns a dictionary indicating the difference for each slider
    between version_b and version_a (i.e. value_b - value_a). Both versions
    must belong to the scenario and the current user's organization.
    """
    import json
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario and org
    cur.execute("SELECT org_id FROM scenarios WHERE id = ?", (scenario_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Fetch both versions
    cur.execute(
        "SELECT version, sliders FROM scenario_versions WHERE scenario_id = ? AND version IN (?, ?)",
        (scenario_id, version_a, version_b),
    )
    rows = cur.fetchall()
    conn.close()
    if len(rows) != 2:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or both versions not found")
    sliders_a = None
    sliders_b = None
    for r in rows:
        if r["version"] == version_a:
            sliders_a = json.loads(r["sliders"]) if r["sliders"] else {}
        elif r["version"] == version_b:
            sliders_b = json.loads(r["sliders"]) if r["sliders"] else {}
    if sliders_a is None or sliders_b is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or both versions not found")
    # Compute slider differences
    diff = {}
    all_keys = set(sliders_a.keys()).union(sliders_b.keys())
    for key in all_keys:
        val_a = float(sliders_a.get(key, 0.0))
        val_b = float(sliders_b.get(key, 0.0))
        diff[key] = val_b - val_a
    # Compute metrics for both versions
    # Compute metrics with calibration for this org
    metrics_a = compute_simulation_metrics(sliders_a, org_id=current_user["org_id"])
    metrics_b = compute_simulation_metrics(sliders_b, org_id=current_user["org_id"])
    metrics_diff = {}
    for mkey in metrics_a.keys():
        metrics_diff[mkey] = metrics_b[mkey] - metrics_a[mkey]
    return {
        "scenario_id": scenario_id,
        "version_a": version_a,
        "version_b": version_b,
        "differences": diff,
        "metrics_differences": metrics_diff,
    }


# -----------------------------------------------------------------------------
# Version Locking and Restore Endpoints
# -----------------------------------------------------------------------------


@app.post("/scenario/{scenario_id}/versions/{version_number}/lock")
def lock_or_unlock_version(
    scenario_id: int,
    version_number: int,
    payload: LockRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Lock or unlock a specific version of a scenario.

    Only users with the role "admin" or "manager" may lock or unlock a
    version. Locking a version marks it as final, preventing accidental
    modifications, but does not prevent creating new versions. Unlocking a
    previously locked version allows it to be considered editable again.
    """
    # Authorize role
    if current_user["role"] not in ("admin", "manager"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only managers or admins may lock/unlock versions")
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario and org
    cur.execute("SELECT org_id FROM scenarios WHERE id = ?", (scenario_id,))
    srow = cur.fetchone()
    if not srow:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if srow["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Verify version exists
    cur.execute(
        "SELECT id, locked FROM scenario_versions WHERE scenario_id = ? AND version = ?",
        (scenario_id, version_number),
    )
    vrow = cur.fetchone()
    if not vrow:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    # Update lock status if it differs
    new_locked = 1 if payload.locked else 0
    cur.execute(
        "UPDATE scenario_versions SET locked = ? WHERE scenario_id = ? AND version = ?",
        (new_locked, scenario_id, version_number),
    )
    conn.commit()
    conn.close()
    # Audit log
    action = "scenario_version_lock" if payload.locked else "scenario_version_unlock"
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        action,
        details=f"{'Locked' if payload.locked else 'Unlocked'} scenario {scenario_id} version {version_number}",
    )
    return {
        "scenario_id": scenario_id,
        "version": version_number,
        "locked": payload.locked,
    }


@app.post("/scenario/{scenario_id}/restore", status_code=status.HTTP_200_OK)
def restore_scenario(
    scenario_id: int,
    payload: RestoreRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Restore a scenario's sliders to those of a specified version.

    The restore operation copies the slider configuration from the given
    version back onto the scenario and also creates a new version to
    record the restoration. This ensures that the previous state is
    preserved and that the restore action is reversible. Users must
    belong to the same organization as the scenario. Any role can
    perform a restore, but the action is logged for auditing.
    """
    import json
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify scenario and org
    cur.execute("SELECT org_id, sliders FROM scenarios WHERE id = ?", (scenario_id,))
    srow = cur.fetchone()
    if not srow:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if srow["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Fetch the requested version
    cur.execute(
        "SELECT sliders, locked FROM scenario_versions WHERE scenario_id = ? AND version = ?",
        (scenario_id, payload.version),
    )
    vrow = cur.fetchone()
    if not vrow:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    # Get sliders data
    sliders_data = vrow["sliders"]
    # Update scenario's sliders and updated_at
    now = _dt.datetime.utcnow().isoformat()
    cur.execute(
        "UPDATE scenarios SET sliders = ?, updated_at = ? WHERE id = ?",
        (sliders_data, now, scenario_id),
    )
    # Determine next version number for restore snapshot
    cur.execute("SELECT MAX(version) as max_version FROM scenario_versions WHERE scenario_id = ?", (scenario_id,))
    mrow = cur.fetchone()
    next_version = (mrow["max_version"] or 0) + 1
    # Insert new version capturing the restored state
    cur.execute(
        "INSERT INTO scenario_versions (scenario_id, version, sliders, created_at, created_by, comment) VALUES (?, ?, ?, ?, ?, ?)",
        (
            scenario_id,
            next_version,
            sliders_data,
            now,
            current_user["id"],
            f"Restored from version {payload.version}",
        ),
    )
    conn.commit()
    conn.close()
    # Audit log
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "scenario_restore",
        details=f"Restored scenario {scenario_id} from version {payload.version} as version {next_version}",
    )
    # Return success and new version info
    return {
        "scenario_id": scenario_id,
        "restored_from": payload.version,
        "new_version": next_version,
    }


################################################################################
# Running the app (for local development)
################################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)