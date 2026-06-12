# f1-pitstop-prediction

Predicting whether an F1 driver will pit within the next 2 laps using quantum machine learning compared against classical baselines. built on Kaggle with FastF1 telemetry (2023 season).

---

## what this actually is

This is a benchmarking study. the goal isn't to prove quantum beats classical (it probably doesn't at this scale), it's to see how a 4-qubit quantum kernel SVM and a variational quantum classifier hold up against XGBoost and LSTM on the same task with the same data. everything runs on a matched subsample so the comparison is fair.

the LLM layer on top (Groq and OpenRouter) generates plain-english strategy explanations for each prediction. so you get a pit probability from the quantum model and then a sentence or two from an LLM explaining what's driving it and whether to act on it.

---

## models compared

| model | notes |
|---|---|
| XGBoost (12 features) | full feature set baseline |
| XGBoost (4 features) | same features the quantum models see |
| LSTM (windowed) | sequential baseline, window size 5 |
| XGBoost (matched subsample) | fair comparison to quantum train size |
| Quantum Kernel SVM | fidelity kernel, 4 qubits, precomputed gram matrix |
| VQC (re-uploading) | angle embedding + StronglyEntanglingLayers, 4 layers |

---

## features

12 features extracted from FastF1 lap data, top 4 selected by mutual information for quantum encoding:

- lap time, sector times (1/2/3)
- tyre life, compound code
- gap to race leader
- 3-lap pace degradation
- position change, lap in stint
- race progress, air temp, track temp

features get scaled and angle-encoded into `[-π, π]` for the quantum circuits.

---

## quantum setup

- 4 qubits, PennyLane `default.qubit`
- kernel: double angle embedding (Y then Z rotation) with CNOT entanglement between each layer
- VQC: re-uploading architecture, 4 ansatz layers, `StronglyEntanglingLayers`, 3 random restarts with early stopping
- training set: 350 samples, test set: 120 samples (matched subsample)

---

## data

Loads real race laps from FastF1 for 5 races: 2023 Bahrain, Spain, Hungary, Italy, Belgium. falls back to a synthetic generator if the API is unavailable (useful for offline testing).

classes are balanced to 800 samples each before train/test split.

---

## LLM strategy layer

After inference, sends the feature values and pit probability to an LLM (tries Groq first, falls back to OpenRouter free tier) with a prompt framed as "you are an F1 strategist." the model returns a 3-sentence read: dominant signal, recommendation (pit/stay/monitor), and main risk of acting on it.

---

## how to run

Open the notebook on Kaggle. you need a Groq or OpenRouter API key stored as a Kaggle secret. everything else installs automatically.

```
pip install fastf1 pennylane pennylane-lightning autoray scikit-learn xgboost torch
```

Cells run top to bottom. kernel matrix computation takes a few minutes (it's O(n²) quantum circuit evaluations). VQC training runs 80 epochs max with early stopping at patience 10.

---

## environment

- Python 3.10+
- PennyLane 0.36.0
- PyTorch (CPU fine, GPU supported)
- FastF1 for telemetry, Groq/OpenRouter for LLM

tested on Kaggle with T4 x2. the quantum parts don't need a GPU since it's simulation, but the LSTM benefits from it.

---

## repo structure

```
f1-pitstop-prediction/
├── f1-pitstop-prediction.ipynb   # everything is here
└── README.md
```

---

## license

MIT
