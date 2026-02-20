# 3D Market Reflexivity & Crash Cascade Simulator

## Creator/Dev
**tubakhxn**

## Project Overview
This project is a fully interactive Streamlit application that simulates nonlinear feedback loops in financial markets. It models how price deviations can trigger volatility, leverage adjustments, forced liquidations, and further price pressure, visualized as a dynamic 3D surface. The app uses:
- Streamlit for UI and controls
- Plotly for 3D visualization
- Numpy, Pandas, Scipy for mathematical modeling

### Features
- Modular simulation of market reflexivity and crash cascades
- Adjustable parameters via sidebar sliders
- Animated 3D surface showing price deviation, leverage, and liquidation intensity
- Numerical stability and clamping to prevent errors

## How to Fork
1. Click the **Fork** button at the top right of the GitHub repository page.
2. Choose your GitHub account as the destination.
3. Clone your forked repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
4. Install dependencies:
   ```bash
   pip install streamlit numpy pandas plotly scipy
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

---
For questions or improvements, contact tubakhxn.
