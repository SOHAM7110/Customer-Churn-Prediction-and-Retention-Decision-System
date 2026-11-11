import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.explain import lr_contributions, top_reasons
from src.features import FEATURES
from src.models import fit_model, DEFAULTS
from tests.test_models import toy

def test_linear_contributions_are_additive():
    X, y = toy()
    pipe = fit_model("lr", DEFAULTS["lr"], "none", X, y)
    sv, base = lr_contributions(pipe, X)
    assert list(sv.columns) == FEATURES
    assert np.allclose(sv.sum(axis=1).values + base, pipe.decision_function(X), atol=1e-8)

def test_top_reasons_orders_by_magnitude():
    X, y = toy()
    pipe = fit_model("lr", DEFAULTS["lr"], "none", X, y)
    sv, _ = lr_contributions(pipe, X)
    r = top_reasons(sv.iloc[0], X.iloc[0], 3)
    mags = [abs(x["shap"]) for x in r]
    assert len(r) == 3 and mags == sorted(mags, reverse=True)
