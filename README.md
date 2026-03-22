# Quantum Machine Learning: QDataSet Analysis

I worked on analyzing quantum systems using machine learning techniques. The project is based on the QDataSet , a collection of quantum datasets designed for training machine learning algorithms on quantum control, tomography, and noise spectroscopy problems.

## Data Source

The dataset is based on the **QDataSet** paper published in Scientific Data (2022). It contains simulations of 1-qubit and 2-qubit systems evolving under different noise conditions.

- **Paper:** Perrier, E., Youssry, A. & Ferrie, C. QDataSet, quantum datasets for machine learning. Sci Data 9, 582 (2022)
- **Source:** https://www.nature.com/articles/s41597-022-01639-1
- **GitHub:** https://github.com/eperrier/QDataSet

The dataset includes:
- 52 datasets with 10,000 samples each
- 1-qubit and 2-qubit systems
- Noise profiles N0 to N6 (Gaussian, non-Gaussian, stationary, non-stationary)
- Control pulses (Gaussian and square)
- Pauli measurement distributions
- V0 operator for noise characterization

## What I Did

I simulated quantum systems based on the QDataSet parameters and analyzed:

- **Quantum State Evolution** — Bloch sphere trajectories for different noise profiles
- **Control Pulses** — Gaussian and square pulse sequences
- **Noise Characterization** — Power spectral density of different noise profiles
- **Pauli Measurements** — Expectation values over time
- **V0 Operator** — Noise characterization using the V0 operator
- **Deep Learning** — Neural network for quantum state classification
- **Quantum Tomography** — State reconstruction from measurements
- **Noise Spectroscopy** — Frequency analysis of quantum noise

## Quantum System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 1.0 | Total time (normalized) |
| M | 1024 | Number of time steps |
| K | 2000 | Number of noise realizations |
| Ω | 12 | Energy gap (1-qubit) |
| Ω₁ | 12 | Energy gap (qubit 1) |
| Ω₂ | 10 | Energy gap (qubit 2) |
| n | 5 | Number of control pulses |
| A_min, A_max | -100, 100 | Amplitude range |
| σ | 8.14e-05 | Pulse spacing std |

## Dataset Categories (Based on QDataSet Table 5)

| Category | Qubits | Drift | Control | Noise |
|----------|--------|-------|---------|-------|
| 1 | 1 | (z) | (x) | (z) |
| 2 | 1 | (z) | (x,y) | (x,z) |
| 3 | 2 | (z1,1z) | (x1,1x) | (z1,1z) |
| 4 | 2 | (z1,1z) | (x1,1x,xx) | (z1,1z) |

## Noise Profiles (N0-N6)

| Profile | Type | Description |
|---------|------|-------------|
| N0 | Noiseless | No noise applied |
| N1 | Colored | 1/f noise with Gaussian bump |
| N2 | Stationary Gaussian | Colored, Gaussian, stationary |
| N3 | Non-stationary Gaussian | Colored, Gaussian, non-stationary |
| N4 | Non-Gaussian | Colored, non-Gaussian, non-stationary |
| N5 | Shifted bump | Similar to N1 with different bump location |
| N6 | Correlated | Correlated to another noise source |

## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 85% (training), 82% (validation) |
| Best Algorithm | Hybrid QNN (96% fidelity) |
| Classical NN | 92% fidelity |
| XGBoost | 88% fidelity |
| SVM | 85% fidelity |

## Quantum Control Framework

| Feature | Symbol | Value | Description |
|---------|--------|-------|-------------|
| Total Time | T | 1.0 | Duration of simulation |
| Time Steps | M | 1024 | Number of discrete steps |
| Noise Realizations | K | 2000 | Number of noise instances |
| Energy Gap | Ω | 12 | Energy splitting of qubit |
| Pulses per Control | n | 5 | Number of control pulses |

## Interactive Dashboard

The dashboard has 12 figures:

| Figure | Title | What It Shows |
|--------|-------|---------------|
| 1.1 | Quantum System Parameters | Hamiltonian & control specifications |
| 1.2 | Dataset Categories | 1-qubit & 2-qubit configurations |
| 1.3 | Control Pulse Sequences | Gaussian & square pulses |
| 1.4 | Noise Profiles (N0-N6) | Quantum noise characterization |
| 1.5 | State Evolution | Bloch sphere trajectories |
| 1.6 | Pauli Measurements | Expectation values over time |
| 1.7 | V₀ Operator Analysis | Noise characterization |
| 1.8 | Deep Learning Model | Architecture for quantum control |
| 1.9 | Training Performance | Loss & accuracy curves |
| 1.10 | Quantum Tomography | State reconstruction |
| 1.11 | Noise Spectroscopy | PSD analysis |
| 1.12 | Benchmark Results | Algorithm comparison |

**Live Dashboard:**  
https://Pratikshat22.github.io/quantum-machine-learning-qdataset/quantum_machine_learning_analysis.html

## What Makes This Different

1. **V₀ Operator Analysis** — I analyzed the V₀ operator which characterizes the effect of noise on quantum measurements. This is not commonly done in student projects.

2. **Noise Profile Comparison** — I compared six different noise profiles (N0-N6) including non-Gaussian and non-stationary noise, which are rarely included in standard quantum ML tutorials.

3. **Bloch Sphere Trajectories** — I visualized quantum state evolution under different control sequences, showing how pulses affect the qubit state.

4. **Quantum Tomography** — I demonstrated state reconstruction from Pauli measurements, a key technique in quantum information.

5. **Power Spectral Density Analysis** — I analyzed the frequency components of quantum noise, which is essential for noise spectroscopy.

6. **Algorithm Benchmark** — I compared classical and hybrid quantum algorithms on fidelity and training time.

## Files

- `quantum_machine_learning_analysis.html` — interactive dashboard
- `analysis_script.py` — Python code
- `README.md` — this file
- `requirements.txt` — dependencies

## Requirements

```bash
pip install -r requirements.txt
