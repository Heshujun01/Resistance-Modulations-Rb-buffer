# Resistance-Modulations-Rb-buffer
access_resistance_project: Comparison of remaining_current_ratio (Ib/I₀) for different analytes in different buffers under varying or identical applied voltages.
# Nanopore Event Analysis: G6P4D11K Dataset

This repository provides a reproducible analysis pipeline for single-channel nanopore blocking events using the **G6P4D11K dataset**. The workflow consists of two Jupyter Notebooks:

1. **`Figure 3 - G6P4D11K(0903).ipynb`**: Processes raw electrophysiology data (`.dat` files) to detect blocking events and compute remaining_current_ratio.
2. **`demo2_20260217.ipynb`**: Loads the processed results and generates key visualizations, including the remaining_current_ratio distribution and its voltage dependence.

All custom logic is encapsulated in the `pySNA.py` module.

> ⚠️ **Note**: Raw data files (e.g., `G6P4D11K_+140mV.dat`) are not included due to size or sensitivity.

------

## 1. System Requirements

### Software Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt
```

### Tested Environment

- **Python**: 3.12
- **OS**: Windows 11 / Ubuntu 22.04
- **Hardware**: Standard desktop (Intel i5+, 16 GB RAM)

### Required Files from User

- One or more `.dat` files from the G6P4D11K experiment series (e.g., recorded at +140 mV, +160 mV, +180 mV).

------

## 2. Installation

```bash
git clone https://github.com/Heshujun01/Resistance-Modulations-Rb-buffer.git
cd Resistance-Modulations-Rb-buffer
pip install -r requirements.txt

```
> **Typical installation time**: ~2 minutes on a standard desktop.

------

## 3. Workflow Execution

### Step 1: Process Raw Data with `Figure 3 - G6P4D11K(0903).ipynb`

1. Place your G6P4D11K `.dat` files in this directory.
2. Open `Figure 3 - G6P4D11K(0903).ipynb` in Jupyter.
3. Update the file paths in the notebook to point to your data (e.g., `"G6P4D11K_+100mV.dat"`).
4. Run all cells.

**Output**: A `.npz` file (e.g., `G6P4D11K_results.npz`) containing:

- `remaining_current_ratio`: Array of Relative remaining current ($I_b / I_0$), calculated as  ($1 - \frac{event\_amplitude}{baseline}$)
- `voltage`: Applied voltage for each event
- Additional metadata (event duration, baseline level, etc.)

**Runtime**: ~1–2 minutes per file.

### Step 2: Visualize Results with `demo2_20260217.ipynb`

1. Open `demo2_20260217.ipynb`.
2. Ensure the `.npz` file from Step 1 is in the same directory.
3. Run all cells.

**Generates**:

- Histogram and kernel density estimate (KDE) of relative remaining current.
- Scatter plot of remaining_current_ratio vs. applied voltage (revealing voltage dependence).

**Runtime**: ~10–30 seconds.

------

## 4. Using Your Own Data

1. Replace the example file paths in `Figure 3 - G6P4D11K(0903).ipynb` with your G6P4D11K `.dat` files.
2. The pipeline automatically handles multiple voltage conditions and groups events accordingly.
3. The output `.npz` file can be used for further statistical analysis or comparison with other datasets (e.g., D10).

------

## 5. Project Structure

```
Resistance-Modulations-Rb-buffer/
├── Figure 3 - G6P4D11K(0903).ipynb  # Raw data processing & event detection
├── demo2_20260217.ipynb             # Visualization of remaining_current_ratio distributions
├── pySNA.py                         # Core analysis functions
├── requirements.txt                 # Exact package versions
└── README.md                        # This file
```
