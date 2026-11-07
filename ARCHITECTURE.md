# System Architecture: Integrated Learning Model (ILM) for Deepfake & Anti-Spoofing Detection in Digital KYC

This document details the architecture, module components, multi-modal feature fusion strategy, risk scoring engine, evaluation methodology, and operational guidelines for the **Integrated Learning Model (ILM)** extension of the `minivision-ai/Silent-Face-Anti-Spoofing` repository.

---

## 1. Overview & Research Goal

Digital Knowledge Your Customer (KYC) verification relies heavily on automated facial recognition. However, facial verification systems are vulnerable to presentation attacks (PAs), including high-resolution printed photos, digital screen replays, 3D silicone masks, and AI-generated deepfakes.

The **ILM Integrated Anti-Spoofing Model** extends the baseline lightweight `MiniFASNet` spatial model into a multi-modal defense framework combining:
1. **Spatial Features**: Deep features from `MiniFASNetV1`/`V2` models.
2. **Temporal Dynamics**: Micro-expression and sequence dynamics via a Gated Recurrent Unit (GRU/LSTM) network.
3. **Physiological Liveness**: Eye Aspect Ratio (EAR), blink count, and head motion variance.
4. **Frequency & Texture Descriptors**: 2D Fast Fourier Transform (FFT) spectral energy distribution and Laplacian texture gradients.
5. **Gated Multi-Modal Feature Fusion**: Dynamic gating mechanism balancing feature modality contributions.
6. **KYC Risk Scoring Engine**: Configurable risk engine evaluating overall risk, risk level (`LOW`, `MEDIUM`, `HIGH`), and decision recommendation (`APPROVE`, `MANUAL REVIEW`, `REJECT`).

> **Note**: This model is inspired by Integrated Learning Model (ILM) concepts for digital KYC security and is not presented as an exact reproduction of any single published paper. The original MiniFASNet spatial backbone models and weights are preserved unchanged as baseline benchmarks.

---

## 2. Architectural Breakdown

```
Input Video Stream / Multi-Frame Sequence
                   │
    ┌──────────────┴──────────────┐
    │  RetinaFace Face Detector   │ (Caffe DNN: Widerface-RetinaFace)
    └──────────────┬──────────────┘
                   │ Bounding Box & Alignment
                   │
    ┌──────────────┼─────────────────────────┬────────────────────────┐
    ▼              ▼                         ▼                        ▼
[Spatial]     [Temporal]                [Liveness]               [Frequency]
MiniFASNet     Multi-Frame GRU         Physiological &          2D FFT Spectrum &
Backbone       Sequence Encoder        Motion Dynamics          Texture Gradients
(128-d)        (64-d)                  (8-d)                    (16-d)
    │              │                         │                        │
    └──────────────┴────────────┬────────────┴────────────────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Gated Feature Fusion    │ (Sigmoid Modal Gates)
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   MLP Classifier Head   │ ──> Integrated Anti-Spoof Probability
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │     KYC Risk Engine     │ ──> Risk Score (0-100), Level, Decision
                   └─────────────────────────┘
```

---

## 3. Module Specifications

### 3.1 Spatial Feature Branch (`src/ilm_model/ilm_spatial.py`)
- **Backbone**: `MiniFASNetV1` / `MiniFASNetV2` / `MiniFASNetV1SE` / `MiniFASNetV2SE`.
- **Function**: Extracts 128-dimensional embedding vector $f_{\text{spatial}} \in \mathbb{R}^{128}$ from face crops across multi-scale patches ($scale=1.0, 2.7, 4.0$).
- **Baseline Integrity**: Uses original pre-trained PyTorch weights (`resources/anti_spoof_models/`).

### 3.2 Temporal Branch (`src/ilm_model/ilm_temporal.py`)
- **Architecture**: Single-layer GRU/LSTM network with temporal attention.
- **Input**: Sequence of spatial vectors $X \in \mathbb{R}^{B \times T \times 128}$ across $T$ frames.
- **Output**: 64-dimensional temporal context embedding $f_{\text{temporal}} \in \mathbb{R}^{64}$ and frame temporal consistency score.

### 3.3 Liveness Feature Branch (`src/ilm_model/ilm_liveness.py`)
- **Inputs**: Frame sequence and bounding box tracking history.
- **Signals Extracted**:
  - Eye Aspect Ratio (EAR) & blink count: $\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$
  - Head motion velocity & variance (bounding box center displacement velocity).
  - Scale/depth variance across consecutive frames.
  - Inter-frame intensity difference & variance.
- **Output**: 8-dimensional liveness vector $f_{\text{liveness}} \in \mathbb{R}^{8}$ and heuristic liveness score $S_{\text{liveness}} \in [0, 1]$.

### 3.4 Frequency & Texture Branch (`src/ilm_model/ilm_frequency.py`)
- **Method**: Reuses 2D Fast Fourier Transform (`generate_FT`) from `src/data_io/dataset_folder.py`.
- **Descriptors Extracted**:
  - Radial power spectrum energy across 4 concentric frequency bands ($B_1$ to $B_4$).
  - High-Frequency Energy Ratio ($\text{HFER} = \frac{\sum E_{\text{high}}}{\sum E_{\text{total}}}$).
  - Laplacian variance (blur / high-frequency attenuation index).
  - Sobel gradient magnitude mean and standard deviation.
  - Spectral entropy ($H_{\text{spec}} = -\sum p_i \log_2 p_i$).
- **Output**: 16-dimensional frequency descriptor $f_{\text{freq}} \in \mathbb{R}^{16}$.

### 3.5 Gated Feature Fusion & Classification (`src/ilm_model/ilm_fusion.py`)
- **Concatenation**: Concatenates raw feature vectors ($128 + 64 + 8 + 16 = 216$-d).
- **Gating Layer**: Learns modal importance scalar gates $[g_s, g_t, g_l, g_f] \in [0, 1]^4$.
- **Fused Representation**: $f_{\text{fused}} = [W_s f_s \cdot g_s \,\|\, W_t f_t \cdot g_t \,\|\, W_l f_l \cdot g_l \,\|\, W_f f_f \cdot g_f] \in \mathbb{R}^{192}$.
- **MLP Head**: Linear(192 -> 96) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(96 -> 32) -> Linear(32 -> 2) -> Softmax.

---

## 4. KYC Risk Engine & Decision Matrix (`src/kyc_risk/kyc_engine.py`)

The KYC Risk Engine integrates multiple authentication signals into a unified Risk Score ($R \in [0, 100]$):

### 4.1 Risk Score Formula
$$\text{Trust Index } S_{\text{trust}} = w_{\text{as}} \cdot S_{\text{antispoof}} + w_{\text{live}} \cdot S_{\text{liveness}} + w_{\text{fm}} \cdot S_{\text{facematch}} + w_{\text{doc}} \cdot S_{\text{doc}}$$

$$\text{Risk Score } R = 100 \times (1.0 - S_{\text{trust}})$$

Default experimental weights (configurable):
- $w_{\text{as}} = 0.40$ (Integrated Anti-Spoofing Probability)
- $w_{\text{live}} = 0.25$ (Liveness Confidence Score)
- $w_{\text{fm}} = 0.25$ (ID Face Matching Cosine Similarity)
- $w_{\text{doc}} = 0.10$ (Document Authenticity Score)

### 4.2 Risk Level Mapping
- **LOW Risk**: $R < 35.0$
- **MEDIUM Risk**: $35.0 \le R < 65.0$
- **HIGH Risk**: $R \ge 65.0$

### 4.3 Action Decision Rules
- **`APPROVE`**: $R < 35.0$ **AND** $S_{\text{antispoof}} \ge 0.70$ **AND** $S_{\text{facematch}} \ge 0.70$
- **`REJECT`**: $R \ge 65.0$ **OR** $S_{\text{antispoof}} < 0.40$ **OR** $S_{\text{facematch}} < 0.40$
- **`MANUAL REVIEW`**: All borderline cases requiring human verification officer review.

---

## 5. Experimental Methodology & Evaluation

Evaluation is performed across 4 comparative configurations using ISO/IEC 30107-3 presentation attack evaluation metrics:

### Evaluated Configurations:
1. **Original MiniFASNet (Baseline)**
2. **MiniFASNet + Temporal**
3. **MiniFASNet + Liveness**
4. **Full ILM Model (Spatial + Temporal + Liveness + Frequency Fusion)**

### Evaluated Metrics:
- **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- **FAR**: False Acceptance Rate (Spoof accepted as Real)
- **FRR**: False Rejection Rate (Real rejected as Spoof)
- **APCER**: Attack Presentation Classification Error Rate
- **BPCER**: Normal Presentation Classification Error Rate
- **ACER**: Average Classification Error Rate ($\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$)
- **Inference Latency**: Per-sequence execution time (ms)

---

## 6. How to Run

### 6.1 Run Baseline Test
```bash
python test.py --image_name image_F1.jpg
```

### 6.2 Run Model Evaluation Suite
```bash
python eval_ilm.py
```
*Generates `evaluation_results.json` and confusion matrix plot images (`confusion_matrix_*.png`).*

### 6.3 Run Interactive KYC Verification Demo
```bash
python demo_kyc.py --id_image_path ./images/sample/image_T1.jpg
```
*Processes webcam feed or falls back to sample simulation, displaying real-time HUD overlays with risk scores and decision badges.*

---

## 7. Limitations & Future Directions

1. **Lightweight Deployment**: Designed for low-latency CPU/edge execution; complex vision transformer temporal models can be explored in future work.
2. **Dataset Scale**: Evaluated on sample verification streams and multi-scale synthetic sequence variations; full-scale benchmark evaluation on CASIA-SURF / SiW datasets recommended for production rollout.
3. **Dynamic Weight Tuning**: Current risk engine weights are set empirically; future extensions could employ Bayesian optimization for risk weight calibration.
