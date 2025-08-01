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