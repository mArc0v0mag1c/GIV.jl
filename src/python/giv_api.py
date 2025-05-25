"""giv_api.py
========================================
Python bridge for **GIV.jl** using *juliacall*.
Place this file in ``src/python`` and import with

```python
from src.python.giv_api import giv, GIVModel
```"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from juliacall import Main as jl

# ---------------------------------------------------------------------------
# Paths & one-time Julia boot
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parents[2]  # ~/GIV.jl repo root
_SRC_DIR = _PROJECT_DIR / "src"                       # contains GIV.jl
_julia_ready = False

def _init_julia() -> None:
    """Boot Julia, activate repo environment, import dependencies (once)."""
    global _julia_ready
    if _julia_ready:
        return

    jl.seval("using Pkg")
    jl.seval(f'Pkg.activate("{_PROJECT_DIR.as_posix()}")')
    jl.seval(f'push!(LOAD_PATH, "{_SRC_DIR.as_posix()}")')
    jl.seval("using DataFrames, StatsModels, Tables, GIV")
    _julia_ready = True

# ---------------------------------------------------------------------------
# Helper – convert guess dict
# ---------------------------------------------------------------------------
def _py_to_julia_guess(guess: dict):
    """Convert Python dict ➜ Julia Dict{String,Float64}."""
    return jl.Dict([(str(k), float(v)) for k, v in guess.items()])

# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------
class GIVModel:
    """Python-native wrapper for Julia GIV results"""

    def __init__(self, jl_model):
        self._jl_model = jl_model

    @property
    def coef(self) -> np.ndarray:
        return np.asarray(self._jl_model.coef)

    @property
    def vcov(self) -> np.ndarray:
        return np.asarray(self._jl_model.vcov)

    @property
    def factor_coef(self) -> np.ndarray:
        return np.asarray(self._jl_model.factor_coef)

    @property
    def factor_vcov(self) -> np.ndarray:
        return np.asarray(self._jl_model.factor_vcov)

    @property
    def agg_coef(self) -> float | np.ndarray:
        agg = self._jl_model.agg_coef
        return float(agg) if jl.isa(agg, float) else np.asarray(agg)

    @property
    def formula(self) -> str:
        return str(self._jl_model.formula)

    def coefficient_table(self) -> pd.DataFrame:
        return coefficient_table(self._jl_model)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def giv(
    df: pd.DataFrame,
    formula: str,
    *,
    id: str,
    t: str,
    weight: Optional[str] = None,
    **kwargs: Any,
) -> GIVModel:
    """Estimate a GIV model from pandas data."""
    _init_julia()

    # Convert inputs
    jdf = jl.DataFrame(df)
    jformula = jl.seval(f"@formula({formula})")
    jid = jl.Symbol(id)
    jt = jl.Symbol(t)
    jweight = jl.Symbol(weight) if weight else jl.nothing

    # Handle keyword arguments
    if isinstance(kwargs.get("algorithm"), str):
        kwargs["algorithm"] = jl.Symbol(kwargs["algorithm"])
    if isinstance(kwargs.get("guess"), dict):
        kwargs["guess"] = _py_to_julia_guess(kwargs["guess"])

    return GIVModel(jl.giv(jdf, jformula, jid, jt, jweight, **kwargs))

# ---------------------------------------------------------------------------
# Coefficient Table Generator
# ---------------------------------------------------------------------------
def coefficient_table(jl_model) -> pd.DataFrame:
    """Get full statistical summary from Julia model"""
    _init_julia()

    # Get Julia's formatted coefficient table
    ct = jl.seval("GIV.coeftable")(jl_model)

    # Extract components
    cols = jl.seval("""
    function getcols(ct)
        cols = [ct.cols[i] for i in 1:length(ct.cols)]
        (; cols=cols, colnms=ct.colnms, rownms=ct.rownms)
    end
    """)(ct)

    # Build DataFrame
    df = pd.DataFrame(
        np.column_stack(cols.cols),
        columns=list(cols.colnms)
    )

    # Add term names if available
    if cols.rownms:
        df.insert(0, "Term", list(cols.rownms))

    # Convert p-values to float
    if "Pr(>|t|)" in df.columns:
        df["Pr(>|t|)"] = df["Pr(>|t|)"].astype(float)

    return df

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example data
    n, T = 4, 6
    rng = np.random.default_rng(0)

    df_example = pd.DataFrame({
        "id": np.repeat(np.arange(1, n+1), T),
        "t": np.tile(np.arange(1, T+1), n),
        "q": rng.standard_normal(n*T),
        "p": np.tile(rng.standard_normal(T), n),
        "w": 1.0
    })

    # Model estimation
    model = giv(
        df_example,
        "q + id & endog(p) ~ 0",
        id="id",
        t="t",
        weight="w",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0}
    )

    # Display results
    print("Estimated coefficients:", model.coef)
    print("\nCoefficient table:")
    print(model.coefficient_table().head())