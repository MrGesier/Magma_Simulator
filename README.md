
# MGTN$ Economy Calibrator

Interactive Streamlit app to calibrate the MGTN$ economy:
- Pool sizing (Primary & Reward)
- Revenue sharing → buybacks
- Reward emission schedule (initial/final rate, half‑life)
- Market depth model (Kaiko k, β) & users/buildings mapping
- DAO cap policy (cadence, EMA trigger, step, floor/ceiling)
- MM loan split across CEXs
- Tier‑1 power‑law ladder calibration with cumulative check
- Revenue editor + monthly/cumulative charts
- Reward pool depletion with buybacks

---

## Quick Start

```bash
# 1) Use a recent Python (3.10+ recommended)
python -V

# 2) Create & activate a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run app_v3.py
