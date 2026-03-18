# 🚀 Ver2.0 Release Notes

Finally got **Ver2.0** together.  
This release is more than just a few fixes — it’s a step toward turning the project into a more complete research framework.

Compared with Ver1.0, Ver2.0 mainly improves the parts I felt were still missing: **more realistic execution logic, better reproducibility, richer analysis tools, and a cleaner engineering structure.**

---

## ✨ What’s new

### 📈 More realistic backtest execution
The execution layer is a lot less idealized now.

Added / improved:

- multiple execution price modes
  - `same_close_idealized`
  - `next_open`
  - `next_close`
- configurable slippage with `slippage_bps`
- clearer transaction cost handling
- execution assumptions are now easier to inspect in outputs

---

### 🧩 Better handling of missing data
For fields like `volume`, `amount`, and `turnover_rate`, this version adds more flexible missing-data handling strategies:

- `zero`
- `ffill`
- `drop`
- `keep_nan`

It can also generate missing-value indicator columns like:

- `volume_was_missing`
- `amount_was_missing`
- `turnover_rate_was_missing`

That makes it easier to tell whether a value was truly zero or filled in later.

---

### 🔁 Stronger reproducibility
I also improved experiment tracking quite a bit in this version.

Runs now record more metadata, including things like:

- git commit
- Python / dependency versions
- data file hashes
- random seed
- timestamps

So it’s much easier to reproduce runs and look back at old results.

---

### 🧠 New research/analysis modules
Ver2.0 adds a few new modules that I plan to keep building on:

- `calibration.py`
- `explainer.py`
- `regime.py`
- `rolling_trainer.py`
- `html_reporter.py`
- `run_manifest.py`
- `repro.py`
- `compare_runs.py`

These are mainly for:

- probability calibration
- model explainability
- regime analysis
- rolling training
- HTML reports
- experiment tracking / replay / comparison

---

## 🛠 Engineering improvements

This release also cleans up the project structure quite a bit:

- added `config_schema.py` for config validation
- added structured logging
- added startup checks
- added a `tests/` suite
- added GitHub Actions CI

So the project feels less like a loose script collection now, and a bit more like something I can keep maintaining properly.

---

## 🐞 Fixes & cleanup

Also fixed a few rough edges along the way, including:

- execution feasibility logic
- listing-day / trading validation issues
- compatibility when `amount` or `volume` is missing
- environment-handling issues during CLI import
- some dependency and CI config problems

---

## ✅ What this release is really about

In simple terms, Ver2.0 is mainly about:

- making backtests less idealized
- making experiments easier to reproduce
- adding deeper research/analysis tools
- making the project easier to extend going forward

---

## 🙌 Final note

Ver1.0 felt more like a working prototype.  
Ver2.0 feels more like the start of a properly shaped research framework.

More improvements will probably come later, and feedback / issues are always welcome.