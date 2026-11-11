PY=python
.PHONY: data labels features baseline train calibrate explain final export app test all
data:      ; $(PY) data/download.py
labels:    ; $(PY) -m src.labels
features:  ; $(PY) -m src.features
baseline:  ; $(PY) -m src.baseline
train:     ; $(PY) -m src.train
calibrate: ; $(PY) -m src.calibrate
explain:   ; $(PY) -m src.explain
# test split is used only below this line, for REPORTING; nothing is selected or tuned from it
final:
	$(PY) -m src.train --final-test
	$(PY) -m src.calibrate --report-test
	$(PY) -m src.decide
	$(PY) -m src.simulate_ab
export:    ; $(PY) -m src.export_app_data
app:       ; streamlit run app/streamlit_app.py
test:      ; $(PY) -m pytest -q tests
all: labels features baseline train calibrate explain final export
