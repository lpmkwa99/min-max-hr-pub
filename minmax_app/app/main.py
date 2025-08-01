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

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi import status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


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


# In‑memory stores
users: Dict[int, User] = {}
tenants: Dict[int, Tenant] = {1: Tenant(id=1, name="Default Tenant")}
scenarios: Dict[int, Scenario] = {}
calibrations: Dict[int, Calibration] = {}
tokens: Dict[str, int] = {}

# ID counters
_user_counter = 1
_scenario_counter = 1
_calibration_counter = 1


################################################################################
# Authentication helpers
################################################################################

security = HTTPBearer(auto_error=False)


def _generate_token() -> str:
    """Generate a random token string."""
    return secrets.token_hex(24)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Retrieve the current authenticated user based on a bearer token.

    This function extracts the token from the Authorization header and
    looks up the corresponding user in the in‑memory token store. If the
    token is missing or invalid, an HTTP 401 error is raised.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    user_id = tokens.get(token)
    if user_id is None or user_id not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return users[user_id]


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
    "Customer Support & CX": {
        "description": "Investments that increase customer satisfaction and retention.",
        "sliders": ["support_staffing", "self_service_tools", "personalization"],
    },
    "Finance & Accounting": {
        "description": "Investments in financial efficiency and reporting accuracy.",
        "sliders": ["automation_tools", "accounts_payable", "strategic_planning"],
    },
    "IT Infrastructure & Security": {
        "description": "Investments in modernization and cybersecurity.",
        "sliders": ["it_modernization", "cybersecurity", "infrastructure_redundancy"],
    },
    "Procurement & Supply Chain": {
        "description": "Investments to optimize inventory and supplier relationships.",
        "sliders": ["inventory_optimization", "supplier_management", "logistics"],
    },
}


################################################################################
# API Endpoints
################################################################################


class UserRegisterRequest(BaseModel):
    """Schema for user registration."""

    username: str
    password: str


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest) -> Dict[str, Any]:
    """Register a new user and return a token.

    This endpoint accepts a JSON body containing a username and password.
    For demonstration purposes the password is stored in plain text. When
    registering, a new user is added to the in‑memory store and a
    bearer token is returned. Subsequent requests should include this
    token in the Authorization header as ``Bearer <token>``.
    """
    global _user_counter
    # Ensure username is unique
    if any(u.username == payload.username for u in users.values()):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    user_id = _user_counter
    _user_counter += 1
    new_user = User(id=user_id, username=payload.username, password=payload.password, roles=["user"], tenant_id=1)
    users[user_id] = new_user
    token = _generate_token()
    tokens[token] = user_id
    return {"token": token, "user_id": user_id}


class UserLoginRequest(BaseModel):
    """Schema for user login."""

    username: str
    password: str


@app.post("/login")
def login(payload: UserLoginRequest) -> Dict[str, str]:
    """Authenticate a user and return an access token."""
    # Simple authentication: match username and password exactly
    for user in users.values():
        if user.username == payload.username and user.password == payload.password:
            # Generate a new token for this session
            token = _generate_token()
            tokens[token] = user.id
            return {"token": token}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)) -> User:
    """Return the profile of the currently authenticated user."""
    return current_user


@app.get("/modules")
def list_modules() -> Dict[str, Any]:
    """List all available modules and their slider definitions."""
    return MODULE_DEFINITIONS


@app.post("/scenario", status_code=status.HTTP_201_CREATED)
def create_scenario(
    payload: SimulationInput,
    current_user: User = Depends(get_current_user),
) -> Scenario:
    """Create a new scenario for the current user.

    The scenario stores the provided slider values and can be
    referenced later for simulations or cloning.
    """
    global _scenario_counter
    scenario_id = _scenario_counter
    _scenario_counter += 1
    now = _dt.datetime.utcnow()
    scenario = Scenario(
        id=scenario_id,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        name=payload.scenario_name or f"Scenario {scenario_id}",
        sliders=payload.sliders,
        created_at=now,
        updated_at=now,
    )
    scenarios[scenario_id] = scenario
    return scenario


@app.get("/scenario/{scenario_id}")
def get_scenario(scenario_id: int, current_user: User = Depends(get_current_user)) -> Scenario:
    """Retrieve a scenario by its ID.

    Users may only access scenarios within their own tenant.
    """
    scenario = scenarios.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if scenario.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return scenario


@app.post("/scenario/{scenario_id}/clone", status_code=status.HTTP_201_CREATED)
def clone_scenario(scenario_id: int, current_user: User = Depends(get_current_user)) -> Scenario:
    """Clone an existing scenario and return the new scenario.

    The cloned scenario copies slider settings and updates the name.
    """
    original = scenarios.get(scenario_id)
    if original is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    if original.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    global _scenario_counter
    new_id = _scenario_counter
    _scenario_counter += 1
    now = _dt.datetime.utcnow()
    clone_name = f"{original.name} (Clone {new_id})"
    cloned = Scenario(
        id=new_id,
        tenant_id=original.tenant_id,
        user_id=current_user.id,
        name=clone_name,
        sliders=original.sliders.copy(),
        created_at=now,
        updated_at=now,
    )
    scenarios[new_id] = cloned
    return cloned


@app.post("/simulate", response_model=SimulationResult)
def run_simulation(
    payload: SimulationInput,
    current_user: User = Depends(get_current_user),
) -> SimulationResult:
    """Run a simulation based on provided slider values.

    The algorithm implemented here is intentionally simplistic. It
    illustrates how slider values might be translated into business
    metrics with linear relationships and diminishing returns. In a
    production setting this function would call into a proper
    simulation engine calibrated with organizational data.
    """
    sliders = payload.sliders
    # Basic metrics initialization
    metrics = {
        "roi": 0.0,
        "employee_satisfaction": 0.0,
        "customer_satisfaction": 0.0,
        "net_profit_margin": 0.0,
        "system_uptime": 0.0,
        "inventory_turnover": 0.0,
    }
    # Apply effects from People module sliders
    people_training = sliders.get("training_budget", 0) / 100.0
    benefits = sliders.get("benefits_budget", 0) / 100.0
    wellness = sliders.get("wellness_programs", 0) / 100.0
    metrics["employee_satisfaction"] += 40 * people_training + 30 * benefits + 20 * wellness
    metrics["roi"] += 5 * (people_training + benefits + wellness)
    # Marketing module
    lead_gen = sliders.get("lead_generation", 0) / 100.0
    social_media = sliders.get("social_media", 0) / 100.0
    events = sliders.get("events", 0) / 100.0
    metrics["roi"] += 60 * lead_gen + 30 * social_media + 10 * events
    # Diminishing returns (quadratic penalty)
    metrics["roi"] -= 10 * ((people_training + benefits + wellness) ** 2 + (lead_gen + social_media + events) ** 2)
    # Customer Support & CX
    support_staffing = sliders.get("support_staffing", 0) / 100.0
    self_service = sliders.get("self_service_tools", 0) / 100.0
    personalization = sliders.get("personalization", 0) / 100.0
    metrics["customer_satisfaction"] += 50 * support_staffing + 30 * self_service + 20 * personalization
    metrics["roi"] += 2 * (support_staffing + self_service + personalization)
    # Finance & Accounting
    automation = sliders.get("automation_tools", 0) / 100.0
    accounts_payable = sliders.get("accounts_payable", 0) / 100.0
    strategic_planning = sliders.get("strategic_planning", 0) / 100.0
    metrics["net_profit_margin"] += 20 * automation + 10 * accounts_payable + 15 * strategic_planning
    metrics["roi"] += 3 * (automation + accounts_payable + strategic_planning)
    # IT Infrastructure & Security
    it_modern = sliders.get("it_modernization", 0) / 100.0
    cybersecurity = sliders.get("cybersecurity", 0) / 100.0
    redundancy = sliders.get("infrastructure_redundancy", 0) / 100.0
    metrics["system_uptime"] += 80 * it_modern + 50 * redundancy - 40 * cybersecurity  # note: cybersecurity lowers incidents but might reduce uptime if overloaded
    metrics["roi"] += 1 * (it_modern + redundancy)  # small ROI improvements
    # Procurement & Supply Chain
    inventory_opt = sliders.get("inventory_optimization", 0) / 100.0
    supplier_mgmt = sliders.get("supplier_management", 0) / 100.0
    logistics = sliders.get("logistics", 0) / 100.0
    metrics["inventory_turnover"] += 50 * inventory_opt + 30 * supplier_mgmt + 20 * logistics
    metrics["net_profit_margin"] += 2 * (inventory_opt + supplier_mgmt + logistics)
    # Normalize and bound metrics between 0 and 100
    for k, v in metrics.items():
        if k == "net_profit_margin":
            # Net profit margin is allowed negative (loss) up to -50%
            metrics[k] = max(min(v, 100.0), -50.0)
        else:
            metrics[k] = max(min(v, 100.0), 0.0)
    messages: List[str] = []
    return SimulationResult(metrics=metrics, messages=messages)


class CalibrationInput(BaseModel):
    """Schema for uploading calibration data.

    ``content`` should be a base64‑encoded string representing the
    contents of the calibration file. ``filename`` is used for
    identification only and is not parsed on upload.
    """

    filename: str
    content: str


@app.post("/calibrate", status_code=status.HTTP_201_CREATED)
def upload_calibration(
    payload: CalibrationInput,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Upload calibration data for the current user's tenant.

    The uploaded file content is provided as a base64 string and is
    stored in memory. No parsing is performed here; the content is
    simply saved. In a production implementation you would parse the
    encoded file and update calibration parameters accordingly.
    """
    import base64

    global _calibration_counter
    try:
        content_bytes = base64.b64decode(payload.content.encode("utf-8"), validate=True)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid base64 content: {exc}")
    calibration_id = _calibration_counter
    _calibration_counter += 1
    calib = Calibration(
        id=calibration_id,
        tenant_id=current_user.tenant_id,
        filename=payload.filename,
        content=content_bytes,
        uploaded_at=_dt.datetime.utcnow(),
    )
    calibrations[calibration_id] = calib
    return {"calibration_id": calibration_id, "filename": payload.filename}


@app.get("/recommendations", response_model=RecommendationResult)
def get_recommendations(
    current_user: User = Depends(get_current_user),
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
    return RecommendationResult(recommendations=recs, rationale=rationale)


@app.get("/hot_swap", response_model=HotSwapResult)
def hot_swap_analysis(
    current_user: User = Depends(get_current_user),
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
    return HotSwapResult(
        missing_capabilities=missing,
        performance_gaps=gaps,
        integration_issues=issues,
        budget_misalignment=budget,
        suggestions=suggestions,
    )


################################################################################
# Running the app (for local development)
################################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)