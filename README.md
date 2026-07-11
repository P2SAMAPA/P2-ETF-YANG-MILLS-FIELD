# Yang-Mills Field Engine for ETFs

Applies Yang-Mills gauge field theory to ETF markets. Capital flows are treated as gauge fields on a principal bundle over the market manifold. The curvature (field strength) identifies arbitrage opportunities as topological obstructions. The per‑ETF score is the field strength at the ETF's position.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- U(1) gauge group (Abelian)
- Connection = returns × macro factor
- Curvature = derivative of connection
- Score = absolute curvature at last time step
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-yang-mills-field-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High field strength → strong gauge field → potential arbitrage / topological obstruction.
- Low field strength → weak gauge field.

## Requirements

See `requirements.txt`.
