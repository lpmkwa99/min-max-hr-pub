"""Initializes the min-max application package.

This package contains the core FastAPI application and in-memory models
required to run the gamified HR/business optimization tool. The package
provides a simple demonstration of the architecture described in the
project specifications, including multi-tenant support, basic user
authentication, scenario management, simulation endpoints, calibration,
recommendation engine stubs, and a hot-swap analysis endpoint.

The implementation in this package is intentionally lightweight and
avoids third-party dependencies beyond FastAPI and uvicorn. It stores
all data in memory rather than a persistent database. Tokens are
generated using Python's ``secrets`` module and are not meant for
production use. Replace these components with proper database models
and secure authentication mechanisms when deploying in a real
environment.
"""

from .main import app  # noqa: F401