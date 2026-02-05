# CGS-Net: Clinical-Guided Spatiotemporal Dual-Configuration Network for Liver Tumor Segmentation 

<center>
  <img src="https://github.com/jylEcho/CGS-Net/blob/main/images/V15.0.png" width="800" alt="">
</center>


## Overview

This repository provides the official implementation of CGS-Net, a clinical-guided spatiotemporal dual-configuration network designed for accurate liver tumor segmentation from contrast-enhanced CT (CECT).
Building upon prior multi-phase fusion strategies, CGS-Net introduces two upgraded configurations — a spatial clinical-guided fusion module and a temporal dual-configuration mechanism — to more effectively capture both inter-phase dependencies and intra-phase feature dynamics.

Unlike existing methods that treat all phases equally, CGS-Net leverages the clinical dominance of the portal venous (PV) phase and explicitly models phase-specific propagation orders, enabling more reliable multi-phase feature integration.

Extensive experiments on PLC-CECT and MPLL datasets demonstrate that CGS-Net consistently outperforms previous approaches (e.g., MW-UNet, MCDA-Net), establishing a new state-of-the-art in both segmentation accuracy and boundary precision.

## Datasets  

| Dataset                                          | Phases               | Disease types  |
| ------------------------------------------------ | -------------------- | -------------- |
| [PLC-CECT](https://github.com/ljwa2323/PLC_CECT) | Multi (NC/ART/PV/DL) | 152,965 slices |
| MPLL                                             | Multi (ART/PV/DL)    | 952,601 slices |

<img src="https://github.com/jylEcho/ICLR26_CSF-Net/blob/main/img/Swin-V5.0.png" width="500">

## 👉 Why Multi-Phase CECT?

Contrast-enhanced CT captures dynamic enhancement patterns via multiple phases:  

- **NC (non-contrast):** baseline anatomy  
- **ART (arterial):** highlights early vascular supply  
- **PV (portal venous):** best lesion–parenchyma contrast  
- **DL (delayed):** reveals washout and boundary refinement 

These phases are complementary, making multi-phase fusion a powerful strategy for robust lesion segmentation.

## Limitations of Existing Fusion Methods

- **Input-level fusion:** simple concatenation, ignores phase importance  
- **Feature-level fusion:** self-attention, but equal weighting across phases  
- **Decision-level fusion:** ensemble of outputs, but lacks inter-phase guidance  

➡️ A key drawback: they treat all phases equally, ignoring **clinical hierarchy (PV > ART > DL)**.  

## ✨ Framework

<center>
  <img src="https://github.com/jylEcho/CSF-Net-plus-plus/blob/main/images/V10.png" width="800" alt="">
</center>


## Our Contributions

- **Systematic single-phase analysis**: On the MPLL dataset, the portal venous (PV) phase demonstrates the strongest segmentation performance, which aligns with its established clinical dominance in liver tumor assessment.
- **Clinically-guided propagation order**: CGS-Net introduces a Multi-Phase Cross-Query Sequential (MCQS) mechanism that explicitly models clinically consistent propagation. Ablation results (Table 3) confirm that the PV→ART→DL order achieves the best performance (76.29% DSC, 61.67% Jaccard), with the lowest HD$_{95}$ and ASSD. 
- **Proposed network:** The upgraded CGS-Net incorporates two enhanced configurations — a spatial clinical-guided fusion module and a temporal dual-configuration strategy — enabling more effective cross-phase feature interaction and fine-grained refinement for robust multi-phase liver tumor segmentation. 

## Evaluation Metrics
We employ several commonly used evaluation metrics to quantitatively evaluate the segmentation performance, including Dice Similarity Coefficient (DSC), Jaccard Similarity Coefficient (JSC), Average Symmetric Surface Distance (ASSD), and 95% Hausdorff Distance (HD$_{95}$) \cite{jiang20253d}.  
For experiments with single-phase (1P), two-phase (2P), and three-phase (3P) inputs, Volume Overlap Error (VOE) and Relative Volume Difference (RVD) are additionally used as supplementary evaluation metrics.

### Dice Similarity Coefficient (DSC)

Since our task involves both **liver** and **tumor** segmentation, the overall DSC is computed as the average of the Dice scores for the liver and tumor regions:

where

$$
\mathrm{DSC} = \frac{2 |A \cap B|}{|A| + |B|}
$$

### Jaccard Similarity Coefficient (JSC) 

$$
\mathrm{JSC} =
w_{\text{liver}} \, \mathrm{JSC}_{\text{liver}}
+
w_{\text{tumor}} \, \mathrm{JSC}_{\text{tumor}}
$$

### Average Symmetric Surface Distance (ASSD)

$$
\mathrm{ASSD}(A, B) =
\frac{1}{|\partial A| + |\partial B|}
\left(
\sum_{x \in \partial A} d(x, \partial B)
+
\sum_{y \in \partial B} d(y, \partial A)
\right)
$$

### 95% Hausdorff Distance (HD$_{95}$)

$$
\mathrm{HD}_{95}(A, B) =
\max \left\{
\mathrm{percentile}_{95}
\bigl( \{ d(x, \partial B) \mid x \in \partial A \} \bigr),
\;
\mathrm{percentile}_{95}
\bigl( \{ d(y, \partial A) \mid y \in \partial B \} \bigr)
\right\}
$$

### Volume Overlap Error (VOE)

$$
\mathrm{VOE} = 1 - \mathrm{JSC}
= 1 - \frac{|A \cap B|}{|A \cup B|}
$$

### Relative Volume Difference (RVD)

$$
\mathrm{RVD} =
\frac{|B| - |A|}{|A|}
$$

### Notation

- $A$: ground-truth segmentation  
- $B$: predicted segmentation  
- $A_c, B_c$: ground-truth and prediction for class $c$  
- $| \cdot |$: volume (number of voxels)  
- $\partial A, \partial B$: surfaces of $A$ and $B$  
- $d(\cdot, \cdot)$: Euclidean distance


## Results

- **On PLC-CECT:** Both configurations of CGS-Net achieve superior results over existing methods.
CGS-Swin attains 76.26% DSC (+1.09%), 62.44% Jaccard (+2.22%), 24.63 HD$_{95}$ (↓4.59), and 14.67 ASSD (↓2.08), surpassing MCDA-Net and MW-UNet by a clear margin.
CGS-CNN also delivers strong performance (76.14% DSC, 61.47% Jaccard), confirming the robustness of the proposed dual-configuration framework across backbone designs.

- **On MPLL:** CGS-Net establishes new state-of-the-art results.
CGS-Swin achieves 76.29% DSC and 61.67% Jaccard, with the lowest HD$_{95}$ and ASSD, while
CGS-CNN attains 75.96% DSC and 61.23% Jaccard, demonstrating consistent cross-dataset generalization and clinical reliability.

- **Qualitative analysis:** Compared to MCDA-Net, CGS-Net produces sharper tumor boundaries (e.g., case 1) and more accurate identification of small nodules (e.g., case 3), validating the effectiveness of its spatiotemporal dual-configuration design for multi-phase feature fusion.

Extensive experiments on two benchmark datasets, LiTS2017 and MPLL, demonstrate the superiority of our proposed method, which significantly outperforms existing state-of-the-art approaches.

## Getting Started

### Before Experiments：Create your conda environment

We recommend using **conda** for a clean and reproducible environment.  

1、environments:Linux 5.4.0

2、conda create -n liverseg python=3.8 -y
conda activate liverseg

3、Install Pytorch : pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

4、Requirements:
numpy==1.14.2
torch==1.0.1.post2
visdom==0.1.8.8
pandas==0.23.3
scipy==1.0.0
tqdm==4.40.2
scikit-image==0.13.1
SimpleITK==1.0.1
pydensecrf==1.0rc3

## 🏋️ Training Pipeline

We provide a step-by-step guide to prepare data and train **OurMethod** for multi-phase liver tumor segmentation. 

### 1️⃣ Dataset Splitting

Split patients into **train / val / test** subsets:  

python multi_phase/multi_phase/dataset_prepare/generate_patients_txt.py

### 2️⃣ Liver Box Generation

Generate liver bounding boxes to accelerate training and reduce background noise:

python multi_phase/multi_phase/dataset_prepare/generate_liverbox.py

📦 Outputs: liver box annotations.

### 3️⃣ Data Preprocessing

Convert raw volumes into 2D slices for training and testing:

# Generate testing slices

python multi_phase/multi_phase/dataset_prepare/rawdata_2D_test.py

# Generate training slices

python multi_phase/multi_phase/dataset_prepare/rawdata_2D_train.py

🖼️ Outputs saved in process_data/train/ and process_data/test/.

### 4️⃣ Train/Test File Lists

Generate .txt files for dataset indexing:

python multi_phase/multi_phase/dataset_prepare/get_txt.py

📂 Outputs:

multi_phase/multi_phase/lists/lists_liver/train.txt

multi_phase/multi_phase/lists/lists_liver/test_vol.txt

### 5️⃣ Start Training

Use the provided training script:

Use the provided training script, Inside train_sformer.sh, training is launched with:

export CUDA_VISIBLE_DEVICES=
cd ..
python train.py \
  --n_gpu 3 \
  --root_path /path to train/ \
  --test_path /path to test/ \
  --module /path to your model/ \
  --dataset Multiphase \
  --eval_interval x # Test every few rounds \
  --max_epochs 100 \
  --batch_size 8 \
  --model_name Fusion \
  --img_size 256 \
  --base_lr 0.01 \

⚙️ Hyperparameters Explained

--n_gpu: number of GPUs to use (e.g., 3 for multi-GPU training).

--root_path: training dataset path.

--test_path: testing dataset path.

--module: network architecture (e.g., HAformerSpatialFrequency).

--dataset: dataset type (here: Multiphase).

--eval_interval: evaluate every N epochs.

--max_epochs: maximum number of training epochs (default 100).

--batch_size: training batch size (e.g., 8).

--model_name: name for saving checkpoints (default: Fusion).

--img_size: input image size (default: 256 × 256).

--base_lr: base learning rate (e.g., 0.01).

### 📊 Outputs

Training checkpoints and logs are saved in:

multi_phase/multi_phase/model_out/

Results include:

✔️ Best model weights

✔️ Training/validation curves

✔️ Evaluation metrics (DSC, Jaccard, HD95, ASSD)

✨ After training, you can run evaluation and visualize results directly from the saved models.













