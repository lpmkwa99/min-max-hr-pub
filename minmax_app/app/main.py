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


# Data models for vendors and recommendations
class Vendor(BaseModel):
    """Represents a service provider in the Solution Provider Index."""
    id: int
    name: str
    category: str
    description: Optional[str]
    cost: float
    rating: Optional[float]
    features: Optional[str]
    integrations: Optional[str]
    impact: Dict[str, float]


class VendorSummary(BaseModel):
    """A summarized view of a vendor for listing purposes."""
    id: int
    name: str
    category: str
    cost: float
    rating: Optional[float]


class RecommendationInput(BaseModel):
    """Schema for requesting tool recommendations.

    Users may specify a budget (total spend available) and a mapping of
    metric names to weights (percentages that sum to 1). An optional
    strategy style can be provided to adjust weights. If omitted, all
    metrics are equally weighted. Examples of metric keys include
    'roi', 'employee_satisfaction', 'customer_satisfaction',
    'productivity_boost', 'churn_rate'. Negative impacts (like
    churn_rate reduction) should be given positive weight if the user
    values reducing churn. The API normalizes weights internally.
    """
    budget: float
    weights: Optional[Dict[str, float]] = None
    strategy_style: Optional[str] = None
    num_results: int = 3


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
    # Streaks table
    # Stores per-user streak state for weekly check-ins. ``streak_count``
    # counts the number of consecutive weeks the user has checked in, and
    # ``last_checkin_date`` stores the date of their most recent check-in.
    # Each row's user_id is a foreign key to the users table. This table
    # supports the "Streak Manager" described in the specification【644198553755745†L638-L677】.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_streaks (
            user_id INTEGER PRIMARY KEY,
            streak_count INTEGER NOT NULL,
            last_checkin_date TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()

    # Vendors table (solution provider index)
    # This table stores information about third‑party service providers/tools
    # available in the Solution Provider Index. Each row represents a vendor
    # with metadata such as category, cost, rating, feature descriptions,
    # integrations, and predicted metric impacts (stored as JSON). This
    # marketplace is mostly global/read‑only; user‑specific actions like
    # bookmarking live in separate tables. See design expansion for details
    #【70839359902819†L338-L373】.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vendors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            cost REAL NOT NULL,
            rating REAL,
            features TEXT,
            integrations TEXT,
            impact_json TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    # Gamification tables
    # Achievements definitions table holds the set of all possible achievements
    # along with their XP rewards. The user_achievements table links users
    # to achievements they have earned and records the timestamp. Storing
    # achievements separately allows us to add new achievements without
    # modifying existing user records and to award XP automatically when
    # achievements are earned【644198553755745†L524-L603】.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS achievements_def (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            xp_reward INTEGER NOT NULL
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_achievements (
            user_id INTEGER NOT NULL,
            achievement_key TEXT NOT NULL,
            achieved_at TEXT NOT NULL,
            PRIMARY KEY (user_id, achievement_key),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (achievement_key) REFERENCES achievements_def(key)
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

# -----------------------------------------------------------------------------
# Organization settings helpers
#
def get_org_settings(org_id: int) -> Dict[str, Any]:
    """Retrieve settings for an organization.

    If no row exists for the given org_id, this function creates one with
    default values (advanced_mode = 0).  Returns a dict with key
    ``advanced_mode`` as an integer (0 or 1).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT advanced_mode FROM org_settings WHERE org_id = ?", (org_id,))
    row = cur.fetchone()
    if not row:
        # create default settings row
        cur.execute(
            "INSERT OR REPLACE INTO org_settings (org_id, advanced_mode) VALUES (?, ?)",
            (org_id, 0),
        )
        conn.commit()
        advanced_mode = 0
    else:
        advanced_mode = row["advanced_mode"]
    conn.close()
    return {"advanced_mode": int(advanced_mode)}

def set_org_advanced_mode(org_id: int, enabled: bool) -> None:
    """Enable or disable advanced mode for a given organization.

    Writes a row into org_settings, creating it if necessary.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO org_settings (org_id, advanced_mode) VALUES (?, ?)",
        (org_id, 1 if enabled else 0),
    )
    conn.commit()
    conn.close()


def init_vendor_data() -> None:
    """Populate the vendors table with sample data if it is empty.

    The Solution Provider Index relies on having a catalog of service providers
    that users can browse and use for recommendations【70839359902819†L338-L373】. On
    startup we check if the table is empty and, if so, insert a handful of
    representative vendors across the major business functions. Each vendor
    includes predicted impacts on key metrics (ROI, engagement, churn, etc.)
    which are used by the recommendation algorithm. In a production system,
    this data would be sourced from APIs like G2 or manually curated and
    updated regularly【70839359902819†L338-L373】.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM vendors")
    row = cur.fetchone()
    if row and row["count"] > 0:
        conn.close()
        return
    # Sample vendors with impact estimates. Impacts are expressed as percentage
    # improvements relative to baseline (positive values increase metrics; for
    # churn reduction negative values indicate reduction in churn rate). These
    # numbers are illustrative and should be refined with real data.
    sample_vendors = [
        {
            "name": "BambooHR",
            "category": "HRIS",
            "description": "All‑in‑one HR platform with employee database, PTO tracking and basic onboarding.",
            "cost": 8.0,
            "rating": 4.5,
            "features": "HRIS, PTO tracking, onboarding",
            "integrations": "Payroll add‑ons, Slack via add‑on, Greenhouse",
            "impact": {"employee_satisfaction": 5.0, "roi": 2.0, "churn_rate": -3.0},
        },
        {
            "name": "Gusto",
            "category": "Payroll",
            "description": "Full‑service payroll and benefits platform with onboarding checklists and compliance.",
            "cost": 6.0,
            "rating": 4.8,
            "features": "Payroll processing, benefits admin, onboarding",
            "integrations": "QuickBooks, Xero, Slack via Zapier",
            "impact": {"employee_satisfaction": 4.0, "roi": 3.0, "churn_rate": -2.0},
        },
        {
            "name": "HubSpot",
            "category": "CRM",
            "description": "CRM and marketing automation platform for sales and marketing alignment.",
            "cost": 50.0,
            "rating": 4.4,
            "features": "Contact management, email marketing, lead scoring",
            "integrations": "Gmail, Slack, Shopify, Salesforce",
            "impact": {"roi": 10.0, "customer_satisfaction": 5.0, "churn_rate": -4.0},
        },
        {
            "name": "Asana",
            "category": "Project Management",
            "description": "Project and task management platform to improve team collaboration and productivity.",
            "cost": 10.0,
            "rating": 4.6,
            "features": "Task tracking, dashboards, integrations",
            "integrations": "Slack, Google Workspace, Zapier",
            "impact": {"productivity_boost": 8.0, "roi": 4.0, "churn_rate": -1.0},
        },
        {
            "name": "QuickBooks Online",
            "category": "Accounting",
            "description": "Cloud accounting platform for small and mid‑sized businesses.",
            "cost": 30.0,
            "rating": 4.7,
            "features": "Bookkeeping, invoicing, reporting",
            "integrations": "Gusto, HubSpot, Shopify",
            "impact": {"net_profit_margin": 3.0, "roi": 2.0},
        },
        {
            "name": "Slack",
            "category": "Collaboration",
            "description": "Team communication tool with channels, messaging and integrations.",
            "cost": 6.0,
            "rating": 4.8,
            "features": "Messaging, file sharing, video calls",
            "integrations": "Gusto, HubSpot, Asana, Zapier",
            "impact": {"productivity_boost": 5.0, "employee_satisfaction": 3.0},
        },
        {
            "name": "Monday.com",
            "category": "Project Management",
            "description": "Work operating system for managing projects, tasks and workflows.",
            "cost": 12.0,
            "rating": 4.5,
            "features": "Boards, automations, dashboards",
            "integrations": "Slack, Google Workspace, Salesforce",
            "impact": {"productivity_boost": 7.0, "roi": 3.0, "employee_satisfaction": 1.0},
        },
    ]
    for vendor in sample_vendors:
        import json
        cur.execute(
            "INSERT INTO vendors (name, category, description, cost, rating, features, integrations, impact_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                vendor["name"],
                vendor["category"],
                vendor["description"],
                vendor["cost"],
                vendor["rating"],
                vendor["features"],
                vendor["integrations"],
                json.dumps(vendor["impact"]),
            ),
        )
    conn.commit()
    conn.close()
    # Organization settings table
    # Holds per‑org configuration flags such as whether advanced mode is enabled.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS org_settings (
            org_id INTEGER PRIMARY KEY,
            advanced_mode INTEGER DEFAULT 0,
            FOREIGN KEY (org_id) REFERENCES organizations(id)
        );
        """
    )
    conn.commit()
    conn.close()

    # Provider lifecycle table
    # Tracks the adoption status of solution providers/tools per organization.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_lifecycle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id INTEGER NOT NULL,
            vendor_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT,
            FOREIGN KEY (org_id) REFERENCES organizations(id),
            FOREIGN KEY (vendor_id) REFERENCES vendors(id)
        );
        """
    )
    conn.commit()
    conn.close()

    # User notifications table
    # Stores messages sent to users (e.g., achievement unlocked notifications) with read status.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            read INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


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


################################################################################
# Gamification Helpers
################################################################################

def init_gamification() -> None:
    """Populate the achievements definition table with default entries.

    This function seeds the database with a set of core achievements if
    none exist. Achievements are defined by a unique key, a user‑friendly
    name, a description, and the amount of XP awarded upon earning them.
    The initial set covers the primary user actions described in the
    implementation plan: creating a scenario, running a simulation,
    uploading calibration data, requesting recommendations, and performing
    a hot swap analysis【644198553755745†L524-L603】. Calling this function
    repeatedly is safe; it inserts definitions only if the table is empty.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS count FROM achievements_def")
    row = cur.fetchone()
    if row and row["count"] > 0:
        conn.close()
        return
    achievements = [
        ("scenario_first", "First Scenario", "Create your first scenario", 50),
        ("simulate_first", "First Simulation", "Run your first simulation", 20),
        ("calibration_first", "First Calibration", "Upload your first calibration dataset", 20),
        ("recommendations_first", "First Recommendations", "Request recommendations for the first time", 20),
        ("hot_swap_first", "First Hot Swap", "Perform your first hot swap analysis", 30),
        ("weekly_streak_4", "Weekly Habit Beginner", "Check in weekly for 4 consecutive weeks", 50),
        ("weekly_streak_8", "Weekly Habit Pro", "Check in weekly for 8 consecutive weeks", 100),
        # ROI achievements reward users for achieving high ROI thresholds
        ("roi_20", "ROI Achiever", "Achieve ROI of at least 20% in a simulation", 40),
        ("roi_50", "ROI Master", "Achieve ROI of at least 50% in a simulation", 80),
    ]
    for key, name, description, xp_reward in achievements:
        cur.execute(
            "INSERT INTO achievements_def (key, name, description, xp_reward) VALUES (?, ?, ?, ?)",
            (key, name, description, xp_reward),
        )
    conn.commit()
    conn.close()


def award_xp(user_id: int, org_id: int, xp_amount: int) -> None:
    """Grant XP to a user and update their level accordingly.

    XP is accumulated on the users table. Levels are computed as
    ``1 + xp // 100``, meaning each 100 XP yields a new level. When
    awarding XP, an audit entry is recorded for traceability. Negative
    or zero amounts are ignored.
    """
    if xp_amount <= 0:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT xp, level FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    current_xp = row["xp"] or 0
    current_level = row["level"] or 1
    new_xp = current_xp + xp_amount
    new_level = 1 + new_xp // 100
    cur.execute("UPDATE users SET xp = ?, level = ? WHERE id = ?", (new_xp, new_level, user_id))
    conn.commit()
    conn.close()
    # Audit event for XP award
    log_audit_event(user_id, org_id, "xp_award", details=f"Awarded {xp_amount} XP; new level {new_level}")


def award_achievement(user_id: int, org_id: int, key: str) -> None:
    """Mark an achievement as earned for a user and award its XP.

    The function checks whether the user already has the specified
    achievement. If not, it inserts a record into ``user_achievements``,
    retrieves the XP reward from ``achievements_def``, and then calls
    ``award_xp`` to grant the XP. An audit entry is created for the
    achievement award. Unknown achievement keys are silently ignored.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Check if user already has achievement
    cur.execute(
        "SELECT 1 FROM user_achievements WHERE user_id = ? AND achievement_key = ?",
        (user_id, key),
    )
    if cur.fetchone():
        conn.close()
        return
    # Retrieve definition
    cur.execute("SELECT xp_reward FROM achievements_def WHERE key = ?", (key,))
    def_row = cur.fetchone()
    if not def_row:
        conn.close()
        return
    xp_reward = def_row["xp_reward"]
    # Insert user achievement
    now = _dt.datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO user_achievements (user_id, achievement_key, achieved_at) VALUES (?, ?, ?)",
        (user_id, key, now),
    )
    conn.commit()
    conn.close()
    # Award XP for the achievement
    award_xp(user_id, org_id, xp_reward)
    # Log achievement award
    log_audit_event(user_id, org_id, "achievement_award", details=f"Achievement {key} awarded")
    # Create a user notification so the UI can display a badge alert
    # The notification table stores a simple message and read flag.
    conn2 = get_db_connection()
    cur2 = conn2.cursor()
    now_ts = _dt.datetime.utcnow().isoformat()
    # Look up the human‑readable achievement name for the message
    cur2.execute("SELECT name FROM achievements_def WHERE key = ?", (key,))
    ach_row = cur2.fetchone()
    ach_name = ach_row["name"] if ach_row else key
    message = f"Congratulations! You earned the achievement '{ach_name}'."
    cur2.execute(
        "INSERT INTO user_notifications (user_id, message, created_at) VALUES (?, ?, ?)",
        (user_id, message, now_ts),
    )
    conn2.commit()
    conn2.close()


def process_gamification_event(user_id: int, org_id: int, event: str) -> None:
    """Handle a gamification event by awarding XP and achievements.

    Events correspond to user actions. The mapping of events to XP
    amounts and achievements is defined inline. If an event is not
    recognized, no action is taken. Achievements are only awarded
    once per user; subsequent occurrences grant XP but skip the
    achievement record if already earned.
    """
    rules = {
        "scenario_create": {"xp": 50, "achievement": "scenario_first"},
        "simulate": {"xp": 10, "achievement": "simulate_first"},
        "calibration_upload": {"xp": 20, "achievement": "calibration_first"},
        "recommendations": {"xp": 20, "achievement": "recommendations_first"},
        "hot_swap": {"xp": 30, "achievement": "hot_swap_first"},
    }
    rule = rules.get(event)
    if not rule:
        return
    xp_amount = rule.get("xp", 0)
    achievement_key = rule.get("achievement")
    if xp_amount:
        award_xp(user_id, org_id, xp_amount)
    if achievement_key:
        award_achievement(user_id, org_id, achievement_key)

    # Special handling for weekly check-ins (streaks)
    if event == "weekly_check_in":
        _update_weekly_streak(user_id, org_id)


def _update_weekly_streak(user_id: int, org_id: int) -> None:
    """Update a user's weekly streak based on the current date.

    A weekly streak increments when the user checks in within 7 days of
    their previous check-in. If more than 7 days have elapsed, the
    streak resets to 1. This function also awards streak-related
    achievements when thresholds are reached (4 and 8 weeks) and grants
    a small XP boost for each check-in. The streak state is stored in
    the ``user_streaks`` table.【644198553755745†L638-L677】
    """
    today = _dt.datetime.utcnow().date()
    conn = get_db_connection()
    cur = conn.cursor()
    # Fetch current streak state
    cur.execute(
        "SELECT streak_count, last_checkin_date FROM user_streaks WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if row:
        last_date = _dt.date.fromisoformat(row["last_checkin_date"])
        streak_count = row["streak_count"]
        # If check-in already recorded today, do nothing
        if last_date == today:
            conn.close()
            return
        days_diff = (today - last_date).days
        if 0 < days_diff <= 7:
            # increment streak
            new_streak = streak_count + 1
        else:
            # reset streak
            new_streak = 1
        # Update row
        cur.execute(
            "UPDATE user_streaks SET streak_count = ?, last_checkin_date = ? WHERE user_id = ?",
            (new_streak, today.isoformat(), user_id),
        )
    else:
        # create new streak entry
        new_streak = 1
        cur.execute(
            "INSERT INTO user_streaks (user_id, streak_count, last_checkin_date) VALUES (?, ?, ?)",
            (user_id, 1, today.isoformat()),
        )
    conn.commit()
    conn.close()
    # Award XP for weekly check-in (5 XP)
    award_xp(user_id, org_id, 5)
    # Award streak achievements
    if new_streak >= 8:
        award_achievement(user_id, org_id, "weekly_streak_8")
    elif new_streak >= 4:
        award_achievement(user_id, org_id, "weekly_streak_4")


# Ensure the database schema is created on module import. This call
# executes the CREATE TABLE statements in init_db() so that all
# subsequent operations against the database can succeed. It is safe
# to call init_db() multiple times because the CREATE TABLE commands
# use IF NOT EXISTS clauses.
init_db()

# Populate the vendors table with initial sample data if needed. This call
# inserts a curated set of service providers for the Solution Provider
# Index. It is safe to call multiple times because it checks whether
# the table already contains entries before inserting new ones.
init_vendor_data()

# Populate the achievements definitions table. Like vendor data, this
# initialization inserts default achievements only if none exist. It
# must be called after ``init_db`` so that the tables are present and
# after ``init_vendor_data`` to ensure the database connection has been
# established. Achievements enable the gamification engine to award
# XP and badges for user actions.
init_gamification()



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

# Advanced module definitions with sub‑levers
#
# The design expansion document introduces an "advanced mode" where each high‑level
# business function is broken down into multiple sub‑levers【70839359902819†L0-L58】.  The
# sliders defined here expose those granular levers for organizations that opt
# into advanced mode via the org_settings table.  Each sub‑lever aligns with
# specific outcomes described in the specifications.  When advanced mode is
# enabled, these definitions replace the standard MODULE_DEFINITIONS for
# purposes of UI display and simulation input.  The simulation combines
# sub‑lever values into the corresponding high‑level domain by averaging them.
ADVANCED_MODULE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "People": {
        "description": "Advanced investment levers for People initiatives.",
        "sliders": [
            "onboarding_efficiency",       # efficiency of onboarding automation
            "engagement_programs",        # employee engagement programs
            "performance_management",     # performance management upgrades
            "leadership_training"          # leadership development investment
        ],
    },
    "Marketing": {
        "description": "Advanced levers for marketing, brand and sales alignment.",
        "sliders": [
            "campaign_quality",           # quality of marketing campaigns
            "brand_development",          # brand development initiatives
            "customer_research",          # investment in market/customer research
        ],
    },
    "Customer Support & CX": {
        "description": "Advanced levers for customer support and experience management.",
        "sliders": [
            "support_staffing",           # staffing level of support team
            "cx_quality_management",      # quality management programs
            "self_service_tools"          # investment in self‑service and automation
        ],
    },
    "Finance & Accounting": {
        "description": "Advanced levers for financial process automation and cost control.",
        "sliders": [
            "process_automation",         # extent of automation of financial processes
            "cost_control",               # investment in cost control initiatives
            "data_accuracy"               # investment in improving financial data accuracy
        ],
    },
    "IT Infrastructure & Security": {
        "description": "Advanced levers for IT modernization and security.",
        "sliders": [
            "infrastructure_modernization",  # modernization of infrastructure
            "cybersecurity_programs",        # cybersecurity programs and tools
            "devops_maturity"                # investment in DevOps maturity and automation
        ],
    },
    "Procurement & Supply Chain": {
        "description": "Advanced levers for procurement and supply chain optimization.",
        "sliders": [
            "inventory_strategy",          # strategy for inventory optimization
            "supplier_negotiations",       # effectiveness of supplier negotiations
            "logistics_efficiency"         # investment in logistics efficiency
        ],
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
    # Gamification: update weekly streak on login
    process_gamification_event(user_id, org_id, "weekly_check_in")
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
def list_modules(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """List all available modules and their slider definitions.

    The set of modules returned depends on whether the caller's organization
    has enabled advanced mode.  In advanced mode the modules expose
    granular sub‑levers as defined in ``ADVANCED_MODULE_DEFINITIONS``【70839359902819†L0-L58】;
    otherwise the base ``MODULE_DEFINITIONS`` are returned.  This endpoint
    requires authentication so that the server can look up the caller's
    organization ID and settings.
    """
    # Determine if advanced mode is enabled for the caller's organization
    org_id = current_user["org_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT advanced_mode FROM org_settings WHERE org_id = ?",
        (org_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row and row["advanced_mode"]:
        return ADVANCED_MODULE_DEFINITIONS
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
    # Gamification: award XP and check achievements
    process_gamification_event(current_user["id"], current_user["org_id"], "scenario_create")
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

    # Determine if advanced mode is enabled for the caller's organization. In
    # advanced mode, the simulation uses sub‑levers from
    # ADVANCED_MODULE_DEFINITIONS and aggregates them into the high‑level
    # domains by averaging the corresponding slider values. This allows
    # organizations to tune granular levers without changing the core
    # simulation formula.
    conn_mode = get_db_connection()
    cur_mode = conn_mode.cursor()
    cur_mode.execute(
        "SELECT advanced_mode FROM org_settings WHERE org_id = ?",
        (current_user["org_id"],),
    )
    row_mode = cur_mode.fetchone()
    advanced_enabled = bool(row_mode["advanced_mode"]) if row_mode else False
    conn_mode.close()

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

    # Helper to aggregate advanced sub‑levers into a single value. Returns the
    # average of provided sliders or 0 if none exist. Assumes inputs are 0–100.
    def avg_sliders(names: List[str]) -> float:
        values = [get_slider(n) for n in names]
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    # 1. People module (training, benefits, wellness)
    if advanced_enabled:
        people_level = avg_sliders(
            ADVANCED_MODULE_DEFINITIONS["People"]["sliders"]
        )
    else:
        training = get_slider("training_budget")
        benefits = get_slider("benefits_budget")
        wellness = get_slider("wellness_programs")
        people_level = (training + benefits + wellness) / 3.0
    people_effect = saturating_effect(people_level)
    base_values["employee_satisfaction"] += 30.0 * people_effect
    base_values["roi"] += 5.0 * people_effect

    # 2. Marketing module (lead generation, social media, events)
    if advanced_enabled:
        # Weighted average for advanced marketing sub‑levers
        # Campaign quality has the largest impact, followed by brand development and research
        msliders = ADVANCED_MODULE_DEFINITIONS["Marketing"]["sliders"]
        # Map order: campaign_quality, brand_development, customer_research
        cq = get_slider("campaign_quality")
        bd = get_slider("brand_development")
        cr = get_slider("customer_research")
        marketing_level = cq * 0.5 + bd * 0.3 + cr * 0.2
    else:
        lead_gen = get_slider("lead_generation")
        social_media = get_slider("social_media")
        events = get_slider("events")
        marketing_level = (lead_gen * 0.6 + social_media * 0.3 + events * 0.1)
    marketing_effect = saturating_effect(marketing_level)
    base_values["roi"] += 50.0 * marketing_effect

    # 3. Customer Support & CX module
    if advanced_enabled:
        # Aggregate advanced levers for support and onboarding
        # support_staffing influences majority, cx_quality_management mid, self_service_tools minor
        ss = get_slider("support_staffing")
        qm = get_slider("cx_quality_management")
        ss_tools = get_slider("self_service_tools")
        support_inv = (ss * 0.5 + qm * 0.3 + ss_tools * 0.2)
        onboarding_quality = (qm * 0.4 + ss_tools * 0.6)  # quality and self‑service drive onboarding
    else:
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
    if advanced_enabled:
        # process_automation, cost_control, data_accuracy
        pa = get_slider("process_automation")
        cc = get_slider("cost_control")
        da = get_slider("data_accuracy")
        automation_level = (pa * 0.5 + da * 0.5)
        cost_opt_level = cc
    else:
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
    if advanced_enabled:
        # infrastructure_modernization, cybersecurity_programs, devops_maturity
        infra = get_slider("infrastructure_modernization")
        cyberprog = get_slider("cybersecurity_programs")
        devops = get_slider("devops_maturity")
        modernization_level = (infra * 0.6 + devops * 0.4)
        cyber_level = cyberprog
    else:
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
    if advanced_enabled:
        # inventory_strategy, supplier_negotiations, logistics_efficiency
        inv = get_slider("inventory_strategy")
        supp = get_slider("supplier_negotiations")
        logi = get_slider("logistics_efficiency")
        inventory_level = (inv * 0.6 + logi * 0.4)
        supplier_level = supp
    else:
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
    # Award ROI achievements based on the final ROI metric
    roi_value = metrics.get("roi", 0.0)
    if roi_value >= 50.0:
        award_achievement(current_user["id"], current_user["org_id"], "roi_50")
    elif roi_value >= 20.0:
        award_achievement(current_user["id"], current_user["org_id"], "roi_20")
    # Audit log – include scenario_name if provided
    scenario_name = payload.scenario_name or "ad‑hoc"
    log_audit_event(current_user["id"], current_user["org_id"], "simulate", details=f"Ran simulation ({scenario_name})")
    # Gamification: award XP and achievements for simulation
    process_gamification_event(current_user["id"], current_user["org_id"], "simulate")
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
    # Gamification: award XP and achievements for calibration upload
    process_gamification_event(current_user["id"], current_user["org_id"], "calibration_upload")
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
# Vendor Marketplace and Recommendations
################################################################################

from typing import Tuple

def fetch_vendors(
    conn: sqlite3.Connection,
    category: Optional[str] = None,
    search: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    min_rating: Optional[float] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
) -> List[sqlite3.Row]:
    """Return a list of vendor rows filtered by the provided criteria.

    This helper executes a SELECT query against the vendors table,
    applying optional filters on category, search substring, cost range and rating.
    Sorting may be requested on cost or rating in ascending or descending order.
    """
    query = "SELECT * FROM vendors WHERE 1=1"
    params: List[Any] = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if search:
        query += " AND (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)"
        term = f"%{search.lower()}%"
        params.extend([term, term])
    if min_cost is not None:
        query += " AND cost >= ?"
        params.append(min_cost)
    if max_cost is not None:
        query += " AND cost <= ?"
        params.append(max_cost)
    if min_rating is not None:
        query += " AND (rating IS NOT NULL AND rating >= ?)"
        params.append(min_rating)
    # Sorting
    if sort_by in {"cost", "rating"}:
        order = "DESC" if sort_order == "desc" else "ASC"
        query += f" ORDER BY {sort_by} {order}"
    # Limit
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    cur = conn.cursor()
    cur.execute(query, tuple(params))
    return cur.fetchall()


@app.get("/vendors", response_model=List[VendorSummary])
def list_vendors(
    category: Optional[str] = None,
    search: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    min_rating: Optional[float] = None,
    limit: int = 20,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[VendorSummary]:
    """List vendors from the Solution Provider Index with optional filters.

    Supports filtering by category, searching by name/description, cost range,
    minimum rating, and sorting by cost or rating. Users must be authenticated,
    but there is no role restriction. Audit entries are recorded.
    """
    conn = get_db_connection()
    rows = fetch_vendors(conn, category, search, min_cost, max_cost, min_rating, limit, sort_by, sort_order)
    conn.close()
    vendors: List[VendorSummary] = []
    for row in rows:
        vendors.append(
            VendorSummary(
                id=row["id"],
                name=row["name"],
                category=row["category"],
                cost=row["cost"],
                rating=row["rating"],
            )
        )
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "vendors_list",
        details=f"Listed vendors (category={category}, search={search})",
    )
    return vendors


@app.get("/vendors/{vendor_id}", response_model=Vendor)
def get_vendor(
    vendor_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Vendor:
    """Retrieve full details for a vendor by ID.

    Returns all metadata including description, features, integrations
    and impact estimates. Raises 404 if not found. Audit entry is recorded.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vendors WHERE id = ?", (vendor_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")
    import json
    impact = json.loads(row["impact_json"]) if row["impact_json"] else {}
    vendor = Vendor(
        id=row["id"],
        name=row["name"],
        category=row["category"],
        description=row["description"],
        cost=row["cost"],
        rating=row["rating"],
        features=row["features"],
        integrations=row["integrations"],
        impact=impact,
    )
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "vendor_detail",
        details=f"Viewed vendor {vendor_id}",
    )
    return vendor


@app.get("/vendor_categories", response_model=List[str])
def list_vendor_categories(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[str]:
    """Return a list of distinct vendor categories.

    This endpoint can be used to build UI filters and navigation. Audit is recorded.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM vendors ORDER BY category ASC")
    categories = [row["category"] for row in cur.fetchall()]
    conn.close()
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "vendor_categories",
        details="Listed vendor categories",
    )
    return categories

# -----------------------------------------------------------------------------
# Vendor comparison and integration network endpoints
# -----------------------------------------------------------------------------

@app.get("/vendors/compare")
def compare_vendors(vendor_ids: str, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Compare multiple vendors on cost and predicted metric impacts.

    Accepts a comma‑separated list of vendor IDs via the ``vendor_ids`` query
    parameter.  Returns a dictionary containing a list of vendor details and
    a comparison matrix of their impacts.  If fewer than two IDs are
    provided, the matrix will be empty.  This endpoint is read‑only and
    available to all authenticated users.  An audit entry is recorded.
    """
    try:
        ids = [int(v.strip()) for v in vendor_ids.split(",") if v.strip()]
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid vendor_ids parameter")
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No vendor IDs provided")
    conn = get_db_connection()
    cur = conn.cursor()
    import json
    placeholders = ",".join(["?"] * len(ids))
    cur.execute(
        f"SELECT id, name, cost, rating, impact_json FROM vendors WHERE id IN ({placeholders})",
        ids,
    )
    rows = cur.fetchall()
    conn.close()
    vendors: List[Dict[str, Any]] = []
    for row in rows:
        impact = json.loads(row["impact_json"]) if row["impact_json"] else {}
        vendors.append({"id": row["id"], "name": row["name"], "cost": row["cost"], "rating": row["rating"], "impact": impact})
    # Build comparison matrix
    matrix: List[Dict[str, Any]] = []
    for i in range(len(vendors)):
        for j in range(i + 1, len(vendors)):
            v1 = vendors[i]
            v2 = vendors[j]
            diff: Dict[str, float] = {}
            metrics = set(v1["impact"].keys()) | set(v2["impact"].keys())
            for m in metrics:
                diff[m] = (v1["impact"].get(m, 0.0)) - (v2["impact"].get(m, 0.0))
            matrix.append({
                "pair": [v1["name"], v2["name"]],
                "cost_diff": v1["cost"] - v2["cost"],
                "impact_diff": diff,
            })
    # Audit event
    log_audit_event(current_user["id"], current_user["org_id"], "vendors_compare", details=f"Compared vendors {vendor_ids}")
    return {"vendors": vendors, "comparisons": matrix}

@app.get("/vendors/integration_network")
def integration_network(vendor_ids: Optional[str] = None, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return an integration network graph for selected vendors.

    The graph consists of nodes (vendors) and edges representing declared
    integrations between vendors.  If ``vendor_ids`` is provided, only those
    vendors are considered; otherwise the entire catalog is used.  An edge
    is created between vendor A and B if A's ``integrations`` string
    contains the name of B (case‑insensitive) or vice versa.  An audit
    entry is recorded.  Clients can use this data to visualize vendor
    compatibility【70839359902819†L414-L457】.
    """
    import json
    conn = get_db_connection()
    cur = conn.cursor()
    rows = None
    if vendor_ids:
        try:
            ids_list = [int(v.strip()) for v in vendor_ids.split(",") if v.strip()]
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid vendor_ids parameter")
        if not ids_list:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No vendor IDs provided")
        placeholders = ",".join(["?"] * len(ids_list))
        cur.execute(
            f"SELECT id, name, integrations FROM vendors WHERE id IN ({placeholders})",
            ids_list,
        )
        rows = cur.fetchall()
    else:
        cur.execute("SELECT id, name, integrations FROM vendors")
        rows = cur.fetchall()
    conn.close()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    # Build node list and dictionary for names
    name_map: Dict[int, str] = {}
    for row in rows:
        nodes.append({"id": row["id"], "name": row["name"]})
        name_map[row["id"]] = row["name"].lower()
    # Determine edges
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = rows[i]
            b = rows[j]
            # Lowercase integration strings
            ints_a = (a["integrations"] or "").lower()
            ints_b = (b["integrations"] or "").lower()
            name_a = name_map[a["id"]]
            name_b = name_map[b["id"]]
            if name_b in ints_a or name_a in ints_b:
                edges.append({"source": a["id"], "target": b["id"]})
    # Audit event
    log_audit_event(current_user["id"], current_user["org_id"], "vendors_integration_network", details=f"Integration network for {vendor_ids or 'all'}")
    return {"nodes": nodes, "edges": edges}


def _compute_baseline_factors(calibrations: Dict[str, float]) -> Dict[str, float]:
    """Compute weighting factors based on calibration baselines.

    For metrics on a 0–100 scale higher is better, the factor is (100 - baseline) / 100
    so that higher baselines reduce the potential benefit. For metrics where
    lower is better (e.g. churn_rate), the factor is baseline / 100 to
    indicate more room for improvement at higher churn. Unknown metrics
    default to 1.0 (no adjustment).
    """
    factors: Dict[str, float] = {}
    for metric in ["employee_satisfaction", "customer_satisfaction", "nps", "churn_rate",
                   "first_response_score", "net_profit_margin", "cogs_reduction",
                   "budget_adherence", "system_uptime", "security_score",
                   "productivity_boost", "inventory_turnover", "procurement_cost_reduction",
                   "lead_time_score", "roi"]:
        baseline_param = f"baseline_{metric}"
        baseline = calibrations.get(baseline_param)
        if baseline is None:
            factors[metric] = 1.0
        else:
            if metric == "churn_rate":
                # Lower churn is better; factor increases with baseline
                factors[metric] = float(baseline) / 100.0
            else:
                factors[metric] = (100.0 - float(baseline)) / 100.0
    return factors

def _recommend_optimal_tools(
    vendors: List[Dict[str, Any]],
    budget: float,
    weights: Dict[str, float],
    calibration_factors: Dict[str, float],
    max_bundle_size: int = 3,
    num_results: int = 3,
) -> List[Tuple[List[str], float, Dict[str, float]]]:
    """Compute optimal combinations of vendors under a budget constraint.

    This helper performs a brute‑force search over all combinations of
    vendors up to ``max_bundle_size`` and returns the top ``num_results``
    based on weighted improvements.  Each vendor dict must include
    ``name``, ``cost`` and ``impact`` keys.  The ``weights`` dict maps
    metric names to user‑defined weights (normalized).  The
    ``calibration_factors`` dict adjusts metric weights based on
    organizational baselines【644198553755745†L1715-L1747】.  The score of a bundle
    is computed as the sum over metrics of (combined impact * weight *
    calibration_factor).  Negative impacts (e.g. churn reduction) are
    treated naturally; combining multiple vendors sums their impacts.
    Returns a list of tuples (list of vendor names, score, combined_impacts)
    sorted in descending score.
    """
    import itertools
    results: List[Tuple[List[str], float, Dict[str, float]]] = []
    n = len(vendors)
    # Precompute vendor attributes for speed
    vendor_impacts = [v["impact"] for v in vendors]
    vendor_costs = [v["cost"] for v in vendors]
    vendor_names = [v["name"] for v in vendors]
    # Generate combinations up to max_bundle_size
    for r in range(1, min(max_bundle_size, n) + 1):
        for combo_indices in itertools.combinations(range(n), r):
            total_cost = sum(vendor_costs[i] for i in combo_indices)
            if total_cost > budget:
                continue
            # Sum impacts across vendors
            combined_impacts: Dict[str, float] = {}
            for i in combo_indices:
                for metric, val in vendor_impacts[i].items():
                    combined_impacts[metric] = combined_impacts.get(metric, 0.0) + val
            # Compute score
            score = 0.0
            for metric, weight in weights.items():
                factor = calibration_factors.get(metric, 1.0)
                impact_val = combined_impacts.get(metric, 0.0)
                score += impact_val * weight * factor
            vendor_list = [vendor_names[i] for i in combo_indices]
            results.append((vendor_list, score, combined_impacts))
    # Sort by score descending and limit to num_results
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:num_results]


@app.post("/recommendations/optimal")
def get_optimal_recommendations(
    payload: RecommendationInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return optimized combinations of vendors based on user weights and budget.

    This endpoint performs a multi‑objective, budget‑constrained search over
    available vendors to identify combinations that maximize weighted metric
    improvements【97530686618726†L0-L145】.  The ``weights`` field in the request
    specifies user priorities for metrics; if absent, equal weights are used.
    Calibration baselines are used to adjust weights such that metrics
    with already‑high baselines contribute less to the score【644198553755745†L1715-L1747】.  The
    ``strategy_style`` field can be one of ``growth``, ``lean`` or
    ``retention``; it modifies default weights accordingly.  The
    ``num_results`` determines how many bundles are returned.  Each bundle
    includes the selected vendor names, total cost, predicted improvements
    and a score.  Audit and gamification events are recorded.
    """
    # Determine weights
    weights = payload.weights or {}
    # Define default weightings by strategy
    default_weights = {
        "roi": 1.0,
        "employee_satisfaction": 1.0,
        "customer_satisfaction": 1.0,
        "productivity_boost": 1.0,
        "churn_rate": 1.0,
    }
    if not weights:
        # If no weights provided, use strategy style or equal weighting
        if payload.strategy_style:
            style = payload.strategy_style.lower()
            if style == "growth":
                default_weights = {
                    "roi": 0.4,
                    "customer_satisfaction": 0.3,
                    "productivity_boost": 0.3,
                }
            elif style == "lean":
                default_weights = {
                    "roi": 0.3,
                    "net_profit_margin": 0.4,
                    "productivity_boost": 0.3,
                }
            elif style == "retention":
                default_weights = {
                    "employee_satisfaction": 0.3,
                    "customer_satisfaction": 0.3,
                    "churn_rate": 0.4,
                }
        weights = default_weights
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {k: 1.0 / len(weights) for k in weights.keys()}
    else:
        weights = {k: v / total_weight for k, v in weights.items()}
    # Fetch calibration factors
    calibrations = get_calibration_params(current_user["org_id"])
    baseline_factors = _compute_baseline_factors(calibrations)
    # Fetch vendors list from DB
    conn = get_db_connection()
    cur = conn.cursor()
    import json
    cur.execute("SELECT id, name, cost, impact_json FROM vendors")
    rows = cur.fetchall()
    conn.close()
    vendors: List[Dict[str, Any]] = []
    for row in rows:
        impact = json.loads(row["impact_json"]) if row["impact_json"] else {}
        vendors.append({"name": row["name"], "cost": row["cost"], "impact": impact})
    # Compute optimal bundles
    combos = _recommend_optimal_tools(vendors, payload.budget, weights, baseline_factors, max_bundle_size=3, num_results=payload.num_results)
    # Format result list
    recommendations: List[Dict[str, Any]] = []
    for vendor_list, score, impacts in combos:
        total_cost = sum(next(v["cost"] for v in vendors if v["name"] == name) for name in vendor_list)
        recommendations.append({"vendors": vendor_list, "score": score, "total_cost": total_cost, "combined_impacts": impacts})
    # Build rationale message for top result
    rationale = "The recommended bundle maximizes your weighted metrics while staying within budget."
    # Audit and gamification events
    log_audit_event(
        current_user["id"], current_user["org_id"], "recommendations_optimal", details=f"Requested optimal recommendations with budget {payload.budget}"
    )
    process_gamification_event(current_user["id"], current_user["org_id"], "recommendations")
    return {"recommendations": recommendations, "rationale": rationale}


@app.post("/recommendations", response_model=RecommendationResult)
def recommend_vendors(
    payload: RecommendationInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RecommendationResult:
    """Provide tool recommendations based on user‑specified weights and budget.

    The recommendation algorithm scores vendors by summing the weighted
    improvements of their metric impacts. Weights are normalized if provided;
    if no weights are given, all metrics present in vendor impacts receive
    equal weight. Calibration baselines adjust the weight of each metric
    so that metrics where the organization is already performing well
    contribute less to the score【644198553755745†L1715-L1747】. Only vendors within the
    specified budget are considered. The top ``num_results`` vendors are
    returned along with a textual rationale.
    """
    # Load vendor catalog
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vendors")
    vendor_rows = cur.fetchall()
    conn.close()
    if not vendor_rows:
        return RecommendationResult(recommendations=[], rationale="No vendors available.")
    # Normalize weights
    weights: Dict[str, float] = {}
    if payload.weights:
        total_weight = sum(abs(w) for w in payload.weights.values()) or 1.0
        for metric, weight in payload.weights.items():
            weights[metric] = abs(weight) / total_weight
    # Retrieve calibration baselines for factor adjustment
    calibrations = get_calibration_params(current_user["org_id"])
    baseline_factors = _compute_baseline_factors(calibrations)
    # Score vendors
    scored: List[Tuple[float, sqlite3.Row]] = []
    import json
    for row in vendor_rows:
        if payload.budget is not None and row["cost"] > payload.budget:
            continue
        impact = json.loads(row["impact_json"]) if row["impact_json"] else {}
        # Determine which metrics to consider
        if weights:
            relevant_metrics = weights.keys()
        else:
            relevant_metrics = impact.keys()
            # assign equal weight to each
            if relevant_metrics:
                weights = {m: 1.0 / len(relevant_metrics) for m in relevant_metrics}
        # compute score
        score = 0.0
        for metric in impact:
            if metric not in weights:
                continue
            improvement = float(impact[metric])
            weight = weights[metric]
            factor = baseline_factors.get(metric, 1.0)
            score += improvement * weight * factor
        # If the vendor has no relevant metrics, skip
        if score == 0.0:
            continue
        # Optionally adjust for cost or rating later; for now just use the improvement
        scored.append((score, row))
    # Sort by score descending
    scored.sort(key=lambda tup: tup[0], reverse=True)
    # Limit to requested number of results
    num = max(1, payload.num_results)
    top = scored[:num]
    recommended_names: List[str] = [row["name"] for _, row in top]
    # Build rationale: highlight which metrics contributed most
    rationale_parts: List[str] = []
    if not top:
        rationale = "No recommendations match your criteria."
    else:
        rationale_parts.append("Top recommendations were selected based on weighted metric improvements and your organization's calibration.")
        for score, row in top:
            impact = json.loads(row["impact_json"]) if row["impact_json"] else {}
            details = []
            for metric, val in impact.items():
                if metric in weights:
                    details.append(f"{metric} {val:+.1f}%")
            rationale_parts.append(f"{row['name']}: {'; '.join(details)}")
        rationale = " ".join(rationale_parts)
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "recommendations",
        details=f"Generated {len(recommended_names)} recommendations within budget {payload.budget}",
    )
    # Gamification: award XP and achievements for requesting recommendations
    process_gamification_event(current_user["id"], current_user["org_id"], "recommendations")
    return RecommendationResult(recommendations=recommended_names, rationale=rationale)


class HotSwapInput(BaseModel):
    """Input schema for performing a hot‑swap analysis on the current tech stack.

    ``current_stack`` should be a list of vendor names currently in use. The
    analysis examines integration gaps, missing capabilities, and potential
    performance improvements based on the organization's calibration data.
    An optional budget can be provided to limit recommended replacements.
    """
    current_stack: List[str]
    budget: Optional[float] = None


@app.post("/hot_swap", response_model=HotSwapResult)
def perform_hot_swap(
    payload: HotSwapInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> HotSwapResult:
    """Analyze the user's current tech stack and suggest additions or replacements.

    The hot‑swap analysis examines the features and impacts of the current
    providers against the organization's calibrated metrics. It identifies
    missing capabilities (categories not represented in the stack), potential
    performance gaps (metrics with low baselines and no vendors improving them),
    simple integration issues (no overlapping integrations), and budget
    misalignment (vendors that are significantly more expensive than peers).
    It then recommends one or two vendors from the index to fill gaps or
    replace underperforming tools【544729577495327†L33-L99】.
    """
    import json
    # Build a set of vendor names from the current stack for lookup
    stack_set = {name.lower() for name in payload.current_stack}
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vendors")
    all_vendors = cur.fetchall()
    conn.close()
    # Determine which vendors are in the stack
    current_vendors: List[sqlite3.Row] = [row for row in all_vendors if row["name"].lower() in stack_set]
    # Identify categories currently covered
    covered_categories = {row["category"] for row in current_vendors}
    # Identify missing categories: categories present in catalog but not in stack
    all_categories = {row["category"] for row in all_vendors}
    missing_categories = sorted(list(all_categories - covered_categories))
    # Identify performance gaps based on calibration baselines
    calibrations = get_calibration_params(current_user["org_id"])
    baseline_factors = _compute_baseline_factors(calibrations)
    # For metrics with factor > 0.5 (indicating significant room for improvement) and
    # no vendor in the stack impacting that metric, mark as gap.
    gaps: List[str] = []
    for metric, factor in baseline_factors.items():
        if factor > 0.5:
            # check if any current vendor improves this metric
            improved = False
            for v in current_vendors:
                impact = json.loads(v["impact_json"]) if v["impact_json"] else {}
                if metric in impact and impact[metric] != 0:
                    improved = True
                    break
            if not improved:
                gaps.append(metric)
    # Identify simple integration issues: if two or more stack tools have no integrations in common
    integration_issues: List[str] = []
    # Build integration sets for each vendor
    vendor_integrations = []
    for v in current_vendors:
        integrations = set()
        if v["integrations"]:
            integrations = {s.strip().lower() for s in v["integrations"].split(",")}
        vendor_integrations.append(integrations)
    # Check pairwise intersection
    for i in range(len(vendor_integrations)):
        for j in range(i + 1, len(vendor_integrations)):
            if vendor_integrations[i] and vendor_integrations[j] and vendor_integrations[i].isdisjoint(vendor_integrations[j]):
                integration_issues.append(
                    f"{current_vendors[i]['name']} and {current_vendors[j]['name']} have no shared integrations"
                )
    # Identify budget misalignment: if any vendor's cost > budget (if provided)
    budget_misalignment: List[str] = []
    if payload.budget is not None:
        for v in current_vendors:
            if v["cost"] > payload.budget:
                budget_misalignment.append(f"{v['name']} exceeds budget")
    # Suggest replacements/additions: for each missing category or gap, pick top vendor impacting that metric
    suggestions: List[str] = []
    # For missing categories, pick highest rated vendor from that category within budget
    for cat in missing_categories:
        # filter vendors by category and cost
        candidates = [row for row in all_vendors if row["category"] == cat]
        if payload.budget is not None:
            candidates = [row for row in candidates if row["cost"] <= payload.budget]
        if not candidates:
            continue
        # sort by rating descending
        candidates.sort(key=lambda r: (r["rating"] or 0.0), reverse=True)
        suggestions.append(f"Add {candidates[0]['name']} to cover category {cat}")
    # For performance gaps, pick vendor with highest impact on that metric
    for metric in gaps:
        # Among vendors not in current stack, find the one with max impact on that metric
        best_row = None
        best_impact = 0.0
        for v in all_vendors:
            if v["name"].lower() in stack_set:
                continue
            impact = json.loads(v["impact_json"]) if v["impact_json"] else {}
            if metric in impact:
                val = float(impact[metric])
                # use absolute improvement for ranking
                if val > best_impact:
                    # consider budget
                    if payload.budget is None or v["cost"] <= payload.budget:
                        best_impact = val
                        best_row = v
        if best_row:
            suggestions.append(f"Consider {best_row['name']} to improve {metric}")
    # Limit suggestions to 5 to avoid overwhelming the user
    suggestions = suggestions[:5]
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "hot_swap",
        details=f"Performed hot swap analysis on stack of {len(payload.current_stack)} vendors",
    )
    # Gamification: award XP and achievements for hot swap
    process_gamification_event(current_user["id"], current_user["org_id"], "hot_swap")
    return HotSwapResult(
        missing_capabilities=missing_categories,
        performance_gaps=gaps,
        integration_issues=integration_issues,
        budget_misalignment=budget_misalignment,
        suggestions=suggestions,
    )


################################################################################
# Gamification Endpoints
################################################################################

@app.get("/xp")
def get_user_xp(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the current user's XP and level.

    XP and level are stored on the users table. This endpoint allows
    users to see their progress in the gamification system. An audit
    entry is recorded when viewed.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT xp, level FROM users WHERE id = ?", (current_user["id"],))
    row = cur.fetchone()
    conn.close()
    xp = row["xp"] if row else 0
    level = row["level"] if row else 1
    # Audit log for viewing XP
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "xp_view",
        details="Viewed XP and level",
    )
    return {"xp": xp, "level": level}


@app.get("/achievements")
def list_achievements(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """List all achievements definitions and indicate which ones the user has earned.

    Returns a list of achievement objects with keys ``key``, ``name``,
    ``description``, ``xp_reward``, ``earned`` (boolean), and
    ``achieved_at`` (ISO timestamp or None). An audit entry is logged.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Fetch definitions
    cur.execute("SELECT key, name, description, xp_reward FROM achievements_def ORDER BY name ASC")
    defs = cur.fetchall()
    # Fetch user's earned achievements
    cur.execute("SELECT achievement_key, achieved_at FROM user_achievements WHERE user_id = ?", (current_user["id"],))
    earned = {row["achievement_key"]: row["achieved_at"] for row in cur.fetchall()}
    conn.close()
    achievements_list: List[Dict[str, Any]] = []
    for row in defs:
        key = row["key"]
        achievements_list.append(
            {
                "key": key,
                "name": row["name"],
                "description": row["description"],
                "xp_reward": row["xp_reward"],
                "earned": key in earned,
                "achieved_at": earned.get(key),
            }
        )
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "achievements_view",
        details="Viewed achievements",
    )
    return achievements_list


@app.get("/leaderboard")
def get_leaderboard(
    scope: str = "org",
    size: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Return a leaderboard of users ranked by XP.

    ``scope`` may be ``org`` to limit the leaderboard to the current
    organization or ``global`` to include all users. ``size`` controls
    the maximum number of users returned (1–100). Results include
    rank, username, XP, and level. An audit entry is recorded when
    viewed.
    """
    # Sanitize size
    size = max(1, min(size, 100))
    conn = get_db_connection()
    cur = conn.cursor()
    if scope == "global":
        cur.execute(
            "SELECT username, xp, level FROM users ORDER BY xp DESC, username ASC LIMIT ?",
            (size,),
        )
    else:
        cur.execute(
            "SELECT username, xp, level FROM users WHERE org_id = ? ORDER BY xp DESC, username ASC LIMIT ?",
            (current_user["org_id"], size),
        )
    rows = cur.fetchall()
    conn.close()
    leaderboard: List[Dict[str, Any]] = []
    rank = 1
    for row in rows:
        leaderboard.append(
            {
                "rank": rank,
                "username": row["username"],
                "xp": row["xp"],
                "level": row["level"],
            }
        )
        rank += 1
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "leaderboard_view",
        details=f"Viewed leaderboard (scope={scope})",
    )
    return leaderboard


@app.get("/streak")
def get_streak(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the current user's weekly streak status.

    Provides the number of consecutive weeks the user has checked in and
    the date of their last check-in. If the user has never checked in,
    returns a streak of 0 and null for the date. An audit entry is
    recorded when viewed.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT streak_count, last_checkin_date FROM user_streaks WHERE user_id = ?",
        (current_user["id"],),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        streak_count = row["streak_count"]
        last_date = row["last_checkin_date"]
    else:
        streak_count = 0
        last_date = None
    # Audit
    log_audit_event(
        current_user["id"],
        current_user["org_id"],
        "streak_view",
        details="Viewed streak status",
    )
    return {"streak_count": streak_count, "last_checkin_date": last_date}

################################################################################
# Organization settings and admin endpoints
################################################################################

class AdvancedModeRequest(BaseModel):
    """Payload for toggling advanced mode."""
    enabled: bool

@app.get("/org/settings")
def get_organization_settings(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return configuration flags for the caller's organization.

    Provides information such as whether advanced mode is enabled.  This
    endpoint is useful for clients to configure their UI accordingly.  It
    records an audit event when accessed.
    """
    settings = get_org_settings(current_user["org_id"])
    # Audit
    log_audit_event(
        current_user["id"], current_user["org_id"], "org_settings_view", details="Viewed org settings"
    )
    return settings

@app.post("/org/advanced_mode", status_code=status.HTTP_200_OK)
def toggle_advanced_mode(payload: AdvancedModeRequest, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Enable or disable advanced mode for the caller's organization.

    Only users with admin or manager roles may change this setting.  When
    advanced mode is enabled, the API will return granular module sliders and
    the simulation engine will aggregate sub‑lever values accordingly.  An
    audit entry is recorded for traceability.
    """
    if current_user["role"] not in {"admin", "manager"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    set_org_advanced_mode(current_user["org_id"], payload.enabled)
    # Audit event
    action = "advanced_mode_enable" if payload.enabled else "advanced_mode_disable"
    log_audit_event(
        current_user["id"], current_user["org_id"], action, details=f"Set advanced_mode to {payload.enabled}"
    )
    return {"advanced_mode": payload.enabled}

# -----------------------------------------------------------------------------
# Administrative endpoints
#
# The following endpoints are only accessible to users with the ``admin`` role.
# They allow an organization to perform user management and organization
# creation.  Admins can create new organizations, invite users, list
# users in their organization, update user roles and delete users.  These
# endpoints enforce tenant isolation, meaning admins cannot manage users
# outside their own organization.

class OrgCreateRequest(BaseModel):
    name: str

@app.post("/admin/organizations", status_code=status.HTTP_201_CREATED)
def create_organization(payload: OrgCreateRequest, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Create a new organization.

    Only admins may create organizations.  The creator's role or org_id is not
    changed by this action.  Returns the new organization's ID and name.  An
    audit entry is recorded for traceability.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO organizations (name) VALUES (?)", (payload.name,))
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization name already exists")
    org_id = cur.lastrowid
    conn.commit()
    conn.close()
    log_audit_event(current_user["id"], current_user["org_id"], "create_organization", details=f"Created org {payload.name}")
    return {"id": org_id, "name": payload.name}


class UserInviteRequest(BaseModel):
    username: str
    password: str
    role: str = Field(default="viewer")
    org_id: Optional[int] = None

@app.post("/admin/users", status_code=status.HTTP_201_CREATED)
def invite_user(payload: UserInviteRequest, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Invite a new user to an organization.

    Admins can add users to their own organization.  Optionally, a different
    organization ID may be specified if the admin manages multiple orgs.  The
    username must be unique.  Returns the new user's ID.  An audit entry is
    recorded.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    org_id = payload.org_id or current_user["org_id"]
    # Validate role
    role = payload.role.lower()
    if role not in {"admin", "manager", "viewer"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
    # Attempt to create the user
    try:
        user_id = create_user(payload.username, payload.password, payload.org_id or get_org_name(org_id), role)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    log_audit_event(current_user["id"], current_user["org_id"], "invite_user", details=f"Invited user {payload.username} with role {role}")
    return {"id": user_id, "username": payload.username, "role": role, "org_id": org_id}

def get_org_name(org_id: int) -> str:
    """Helper to fetch an organization's name by ID."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM organizations WHERE id = ?", (org_id,))
    row = cur.fetchone()
    conn.close()
    return row["name"] if row else "Unknown"

@app.get("/admin/users")
def list_org_users(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """List users in the admin's organization.

    Admins and managers may list users in their own organization.  The
    response includes user IDs, usernames and roles but not password hashes.
    """
    if current_user["role"] not in {"admin", "manager"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, role FROM users WHERE org_id = ?", (current_user["org_id"],))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.put("/admin/users/{user_id}")
def update_user_role(user_id: int, role: str, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Update the role of a user within the admin's organization.

    Only admins may change roles.  The user must belong to the same
    organization as the admin.  Valid roles are admin, manager, viewer.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    role = role.lower()
    if role not in {"admin", "manager", "viewer"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
    conn = get_db_connection()
    cur = conn.cursor()
    # Check user belongs to same org
    cur.execute("SELECT org_id FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify user in another organization")
    cur.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    conn.close()
    log_audit_event(current_user["id"], current_user["org_id"], "update_user_role", details=f"Updated user {user_id} role to {role}")
    return {"id": user_id, "role": role}

@app.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> None:
    """Delete a user from the admin's organization.

    Only admins may delete users.  Deleting a user also removes their
    sessions, streaks and user achievements.  The admin cannot delete
    themselves.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    if user_id == current_user["id"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete yourself")
    conn = get_db_connection()
    cur = conn.cursor()
    # Verify user exists and belongs to same org
    cur.execute("SELECT org_id FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if row["org_id"] != current_user["org_id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete user in another organization")
    # Remove user records
    cur.execute("DELETE FROM user_achievements WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM user_streaks WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    log_audit_event(current_user["id"], current_user["org_id"], "delete_user", details=f"Deleted user {user_id}")
    return None

################################################################################
# Provider lifecycle management
################################################################################

class LifecycleRequest(BaseModel):
    vendor_id: int
    status: str = Field(..., description="Lifecycle status: planning, adopted, pilot, deprecated")
    end_date: Optional[str] = None

@app.post("/lifecycle", status_code=status.HTTP_201_CREATED)
def add_or_update_lifecycle(payload: LifecycleRequest, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Add or update a vendor lifecycle entry for the caller's organization.

    The status must be one of: planning, pilot, adopted, deprecated.  If a
    lifecycle entry for the vendor/org combination does not exist, it is
    created with the current date as the start_date.  If it does exist, the
    status is updated and end_date may be set when status becomes deprecated.
    Only managers and admins may modify lifecycle entries.
    """
    if current_user["role"] not in {"admin", "manager"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    status_lower = payload.status.lower()
    valid_status = {"planning", "pilot", "adopted", "deprecated"}
    if status_lower not in valid_status:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid lifecycle status")
    conn = get_db_connection()
    cur = conn.cursor()
    # Check existing entry
    cur.execute(
        "SELECT id, status FROM provider_lifecycle WHERE org_id = ? AND vendor_id = ?",
        (current_user["org_id"], payload.vendor_id),
    )
    row = cur.fetchone()
    now_str = _dt.datetime.utcnow().isoformat()
    if row:
        lifecycle_id = row["id"]
        cur.execute(
            "UPDATE provider_lifecycle SET status = ?, end_date = ? WHERE id = ?",
            (status_lower, payload.end_date, lifecycle_id),
        )
    else:
        # Insert new entry
        cur.execute(
            "INSERT INTO provider_lifecycle (org_id, vendor_id, status, start_date, end_date) VALUES (?, ?, ?, ?, ?)",
            (current_user["org_id"], payload.vendor_id, status_lower, now_str, payload.end_date),
        )
        lifecycle_id = cur.lastrowid
    conn.commit()
    conn.close()
    log_audit_event(
        current_user["id"], current_user["org_id"], "provider_lifecycle_update", details=f"Vendor {payload.vendor_id} status {status_lower}"
    )
    return {"id": lifecycle_id, "vendor_id": payload.vendor_id, "status": status_lower}

@app.get("/lifecycle")
def list_lifecycle(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """List all lifecycle entries for the caller's organization."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT pl.id, pl.vendor_id, v.name as vendor_name, pl.status, pl.start_date, pl.end_date FROM provider_lifecycle pl "
        "JOIN vendors v ON pl.vendor_id = v.id WHERE pl.org_id = ?",
        (current_user["org_id"],),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

################################################################################
# User notifications
################################################################################

@app.get("/notifications")
def list_notifications(all: bool = False, current_user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """Return notifications for the current user.

    By default only unread notifications (read = 0) are returned.  Pass
    ``?all=true`` to include both read and unread notifications.  Each
    notification includes its ID, message, timestamp and read flag.  The
    user may later mark notifications as read via the `/notifications/{id}/read`
    endpoint.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    if all:
        cur.execute(
            "SELECT id, message, created_at, read FROM user_notifications WHERE user_id = ? ORDER BY created_at DESC",
            (current_user["id"],),
        )
    else:
        cur.execute(
            "SELECT id, message, created_at, read FROM user_notifications WHERE user_id = ? AND read = 0 ORDER BY created_at DESC",
            (current_user["id"],),
        )
    rows = cur.fetchall()
    conn.close()
    # Audit
    log_audit_event(current_user["id"], current_user["org_id"], "notifications_view", details="Viewed notifications")
    return [dict(row) for row in rows]

@app.post("/notifications/{notification_id}/read", status_code=status.HTTP_200_OK)
def mark_notification_read(notification_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Mark a notification as read for the current user."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure the notification belongs to the user
    cur.execute(
        "SELECT user_id, read FROM user_notifications WHERE id = ?",
        (notification_id,),
    )
    row = cur.fetchone()
    if not row or row["user_id"] != current_user["id"]:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    cur.execute("UPDATE user_notifications SET read = 1 WHERE id = ?", (notification_id,))
    conn.commit()
    conn.close()
    # Audit
    log_audit_event(
        current_user["id"], current_user["org_id"], "notification_read", details=f"Marked notification {notification_id} as read"
    )
    return {"id": notification_id, "read": True}


################################################################################
# Running the app (for local development)
################################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)