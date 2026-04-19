# ◎ TribeV2 Neural Intelligence Engine

<p align="center">
  <img src="dashboard/public/tribev2_demo.webp" alt="TribeV2 Dashboard Overview" style="width: 100%; border-radius: 8px;" />
</p>


**TribeV2** is a high-fidelity, deterministic neuromarketing analytics dashboard. It simulates and visualizes predicted human neural activation during video stimulus, leveraging a simulated cognitive data pipeline. The application fuses responsive React UI with a real-time 3D WebGL cortical rendering system to provide deep-tech insights into audience engagement.

## 🚀 Core Features

- **Neuro-Digital Twin**: A high-performance Three.js 3D cortical mesh mapped to the Yeo 7-network functional atlas, calculating localized neural activation (e.g., Dorsal Attention, Limibic) in real-time.
- **Dynamic MP4 Upload Engine**: A local-first synchronized video stimulus block. Instantly map custom video inputs against predictive neural timelines without backend latency.
- **Demographic Dynamics Tracking**: Comparative line charts isolating cohort responses (Gen Z, Adults, Kids, Older) across audio/visual stimulus events.
- **Automated High-Impact Scene Extraction**: Algorithmically detects and splices video thumbnails corresponding to the exact moments of peak global neural activation.
- **Regional Variance Analysis**: Deep-dive analytics into Regions of Interest (ROIs like V4t, MT, TPOJ3) measuring inter-subject consistency vs. fluctuation.

## 🛠️ Architecture

*   **Frontend**: React, Vite, Three.js (`@react-three/fiber`, `@react-three/drei`), Recharts.
*   **Data Structure**: Deterministic JSON-based timeline generation allowing agent-less, zero-API dependency scaling.
*   **Aesthetic**: Ad-agency dark mode UI, utilizing exact HEX palette targeting and spatial design language.

## 💻 Running Locally

1. **Navigate to Dashboard Directory**
   ```bash
   cd dashboard
   ```
2. **Install Dependencies**
   ```bash
   npm install
   ```
3. **Launch Data Viz Engine**
   ```bash
   npm run dev
   ```

## 🧠 Data Generation (Python)

To re-build the deterministic neural simulation mapping (for new timestamps or variants):
```bash
python generate_full_data.py
```
This will output a `brain_data_full.json` directly into the Vite `public/` directory for immediate React consumption.
