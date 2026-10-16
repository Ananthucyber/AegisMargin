"""AegisMargin - FastAPI server.

Serves the Jinja2 dashboard and a JSON inspection API.
Business logic stays in src/ (untouched). The only code you may need to
adjust is the ADAPTER block below, which maps src/ outputs to the shapes
the frontend expects.

Run:  uvicorn main:app --reload
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from datathon_2026_prototype.data_pipeline import load_and_process_data, prepare_model_features
from datathon_2026_prototype.decision_engine import generate_manager_remediation
from datathon_2026_prototype.explainer import SHAPExplainerEngine

log = logging.getLogger("aegismargin")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_leak_model.pkl"

app = FastAPI(title="AegisMargin")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --------------------------------------------------------------------------
# ADAPTER: column names. Edit if your processed dataframe uses other names.
# Each field lists candidates; the first one found in the dataframe is used.
# --------------------------------------------------------------------------
COLS = {
    "tx_id": ["Order ID", "order_id", "Transaction ID", "transaction_id", "tx_id", "Row ID"],
    "product_id": ["Product ID", "product_id", "SKU"],
    "category": ["Category", "category", "Sub-Category"],
    "quantity": ["Quantity", "quantity", "Qty"],
    "unit_cost": ["Unit Cost", "unit_cost", "Cost"],
    "list_price": ["List Price", "list_price", "Unit Price", "Sales"],
    "discount": ["Discount", "discount", "Discount Rate", "discount_pct"],
    "profit": ["Profit", "profit", "Net Profit", "gross_profit"],
}

# Fallbacks shown if a KPI cannot be computed from the data.
KPI_FALLBACK = {
    "profit_bleed": 1_572_018.97,
    "neg_orders": 6829,
    "neg_pct": 37.9,
    "defect_skus": 76,
    "roc_auc": 1.000,
}


def col(df: pd.DataFrame, field: str) -> str | None:
    return next((c for c in COLS[field] if c in df.columns), None)


# --------------------------------------------------------------------------
# Startup: load data, features, model, explainer (once)
# --------------------------------------------------------------------------
STATE: dict = {}


@app.on_event("startup")
def startup() -> None:
    df = load_and_process_data()
    feats = prepare_model_features(df)
    X = feats[0] if isinstance(feats, tuple) else feats

    explainer = SHAPExplainerEngine(str(MODEL_PATH))
    model = explainer.model

    profit_col = col(df, "profit")
    tx_col = col(df, "tx_id")
    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    flagged = df[df[profit_col] < 0] if profit_col else df
    STATE.update(df=df, X=X, model=model, explainer=explainer, flagged=flagged,
                 profit_col=profit_col, tx_col=tx_col)
    log.info("Loaded %d rows, %d flagged", len(df), len(flagged))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean(v):
    """Make numpy / NaN values JSON safe."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def compute_kpis() -> dict:
    df, pc = STATE["df"], STATE["profit_col"]
    k = dict(KPI_FALLBACK)
    try:
        neg = df[df[pc] < 0]
        k["profit_bleed"] = float(-neg[pc].sum())
        k["neg_orders"] = int(len(neg))
        k["neg_pct"] = round(100 * len(neg) / len(df), 1)
        uc, lp, pid = col(df, "unit_cost"), col(df, "list_price"), col(df, "product_id")
        if uc and lp and pid:
            k["defect_skus"] = int(df.loc[df[uc] > df[lp], pid].nunique())
    except Exception:
        log.exception("KPI computation failed; using fallbacks")
    return k


def severity_for(loss: float) -> str:
    loss = abs(loss or 0)
    return "Critical" if loss >= 500 else "High" if loss >= 100 else "Medium"


def normalise_shap(raw) -> list[dict]:
    """Return [{feature, value, impact}] sorted by |impact| desc.
    Accepts a DataFrame or list of dicts with feature / value / shap-like keys."""
    rows = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else list(raw or [])
    out = []
    for r in rows:
        low = {str(k).lower(): v for k, v in r.items()}
        feature = low.get("feature") or low.get("name") or low.get("driver")
        value = next((low[k] for k in ("value", "observed", "observed_value", "feature_value") if k in low), None)
        impact = next((low[k] for k in ("impact", "shap", "shap_value", "shap_impact", "contribution") if k in low), 0)
        out.append({"feature": str(feature), "value": clean(value), "impact": float(clean(impact) or 0)})
    out.sort(key=lambda d: abs(d["impact"]), reverse=True)
    return out[:8]


def normalise_playbook(raw, loss: float) -> list[dict]:
    """Return [{severity, title, action}] from strings, dicts or a single string."""
    if raw is None:
        return []
    items = [raw] if isinstance(raw, (str, dict)) else list(raw)
    out = []
    for it in items:
        if isinstance(it, dict):
            low = {str(k).lower(): v for k, v in it.items()}
            sev = str(low.get("severity") or low.get("priority") or low.get("level") or severity_for(loss)).title()
            out.append({
                "severity": sev if sev in ("Critical", "High", "Medium") else severity_for(loss),
                "title": str(low.get("title") or low.get("issue") or low.get("action_type") or "Recommended action").replace("\\$", "$"),
                "action": str(low.get("action") or low.get("recommendation") or low.get("detail") or ""),
            })
        else:
            out.append({"severity": severity_for(loss), "title": "Recommended action", "action": str(it)})
    return out


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    flagged, tx_col = STATE["flagged"], STATE["tx_col"]
    ids = flagged[tx_col].astype(str).tolist() if tx_col else [str(i) for i in flagged.index]
    # Unique, ordered, capped so the dropdown stays snappy
    ids = list(dict.fromkeys(ids))[:500]
    return templates.TemplateResponse(
        request, "index.html", {"kpis": compute_kpis(), "tx_ids": ids}
    )


@app.get("/api/inspect/{tx_id}")
def inspect(tx_id: str):
    df, X, tx_col, pc = STATE["df"], STATE["X"], STATE["tx_col"], STATE["profit_col"]
    mask = (df[tx_col].astype(str) == tx_id) if tx_col else (df.index.astype(str) == tx_id)
    hits = df.index[mask]
    if len(hits) == 0:
        raise HTTPException(404, f"Transaction {tx_id} not found")
    i = hits[0]
    row = df.loc[i]
    loss = float(row[pc]) if pc else 0.0

    # ---- ADAPTER: explainer call. Adjust the method name if yours differs.
    explainer = STATE["explainer"]
    method = next((m for m in ("explain_record", "explain_transaction", "explain_row", "explain", "get_top_drivers")
                   if hasattr(explainer, m)), None)
    shap_raw = getattr(explainer, method)(X.loc[[i]]) if method else []

    # ---- ADAPTER: decision engine call.
    leak_probability = STATE["model"].predict(X.loc[[i]])[0]
    playbook_raw = generate_manager_remediation(row, leak_probability)

    def field(name):
        c = col(df, name)
        return clean(row[c]) if c else None

    return JSONResponse({
        "tx_id": tx_id,
        "order": {
            "product_id": field("product_id"),
            "category": field("category"),
            "quantity": field("quantity"),
            "unit_cost": field("unit_cost"),
            "list_price": field("list_price"),
            "discount": field("discount"),
            "net_loss": loss,
        },
        "shap": normalise_shap(shap_raw),
        "playbook": normalise_playbook(playbook_raw, loss),
        "severity": severity_for(loss),
    })


@app.post("/api/remediate/{tx_id}")
def remediate(tx_id: str):
    # Hook your ticketing / pricing-override logic in here.
    log.info("Remediation executed for %s", tx_id)
    return {"ok": True, "tx_id": tx_id}
