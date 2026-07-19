# Federated Learning for Skin Lesion Classification
## Project Report & Analysis

### 1. Problem Statement
With the rapid increase in skin cancer cases globally, robust computer-aided diagnostic (CAD) systems are essential. However, training highly accurate deep learning models for clinical diagnostics strictly requires vast amounts of sensitive patient data. Centralizing cross-institutional medical parameters poses significant data-privacy constraints, and standard anonymization is fundamentally insufficient.

This project investigates **Federated Learning (FL)** as an alternative framework capable of securely training a decentralized MobileNetV3 classification system. Specifically, we evaluate whether FL can maintain competitive macro-F1 and AUC scores against traditional centralized models, whilst simulating real-world hospital challenges such as Non-IID (Independent and Identically Distributed) data imbalances and distinct domain shifts.

### 2. Methodology
To conduct this comparative machine learning study, the workflow is isolated into dual execution strategies sharing a universal parameter architecture:

**Dataset Selection:** 
*   **HAM10000:** The localized environment uses exactly 10,015 stratified dermoscopic images spanning 7 distinct demographic lesion classifications.
*   **ISIC Extension:** A secondary distribution subset to mathematically measure adaptation to 'domain shifts' (simulating the inclusion of a totally separate hospital clinic).

**Model Architecture (MobileNet + Attention):**
*   We utilize a high-efficiency `MobileNetV3-Large` backbone targeting CPU-bound client edges. 
*   To prioritize critical diagnostic structures without inflating parameter counts, a dual-layer **Squeeze-and-Excitation (SE)** spatial attention block was dynamically hardcoded preceding the dense classifier layer.

**Federated System (Flower - `flwr`):**
*   Using a centralized `FedAvg` (Federated Averaging) aggregator logic, virtual clients simulate independent clinic nodes, maintaining their data completely localized.
*   **Distribution Simulation** is processed identically inside `fl/simulation.py`. The framework controls distinct environment flags (IID, Non-IID, Extreme Non-IID) that intentionally silo and unbalance the labels to force model drift evaluation.

### 3. Execution & Results
**Centralized Baseline (Upper Bound):**
When the model executes the entire holistic dataset asynchronously (`centralized_train.py`), the convergence logic aggressively forces the AUC toward an optimal baseline threshold. Utilizing mixed mixup augmentations and cosine learning slopes, the gradient establishes an upper mathematical ceiling for identical generalized prediction.

**Federated Generalization:**
When shifted onto the Federated pipeline simulating standard communication rounds, training consistently proves slower due to localized gradients clashing during the generic `FedAvg` averaging sequence. 
As confirmed by the Matplotlib visual mapping generated via `compare.py`:
*   **IID Performance:** Trajectories are closely tethered to centralized baselines.
*   **Non-IID Divergence:** Validation variance across `Client 0`, `Client 1`, and `Client 2` violently separates under extreme imbalance protocols. Client 2 inherently forgets historical states due to missing label distributions. 

**Grad-CAM Visualizations Layer:**
We hooked an interception extraction script directly against the final functional convolution block using `gradcam.py`. The resulting heatmaps accurately visualize bounding-box derivations. The MobileNet visually concentrates localized gradient attention across exact lesion borders, effectively bypassing background noise features.

### 4. Conclusion
Our experiments successfully validate that **Federated Learning resolves diagnostic data constraints without completely sacrificing predictive integrity**. 

While highly imbalanced label distributions (Extreme Non-IID scenarios) strictly penalize macro-F1 metrics temporarily due to localized bias shifts, the global Federated layer forcibly maintains high-threshold AUC convergence compared to any model attempting to train strictly inside a local, isolated hospital silo. By dynamically augmenting image arrays and incorporating custom attention blocks across decentralized nodes, clinics can collaborate via shared updates, ensuring algorithmic integrity strictly while conforming perfectly to secure data-privacy barriers.
