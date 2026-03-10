# Synthetic Medical X-Ray Generation & Classification Pipeline
[cite_start]**Course:** CSET419 - Introduction to Generative AI [cite: 1]  
[cite_start]**Experiment:** Lab 1 - Implement a Simple Generative Algorithm for Data Generation [cite: 2, 4]

## 📌 Project Overview
This project demonstrates the fundamentals of Generative AI by creating a synthetic medical imaging dataset and using it to train a diagnostic classifier. The workflow consists of two main stages:
1.  [cite_start]**Data Generation:** Using a pre-trained **Stable Diffusion** model to generate synthetic chest X-rays based on medical text prompts[cite: 5, 9, 11].
2.  **Classification:** Training a **ResNet-18** Deep Learning model (Transfer Learning) to classify these synthetic images into 10 distinct medical categories.

## 📂 Dataset Structure
[cite_start]The generated dataset is stored in a structured folder format suitable for `torchvision.datasets.ImageFolder`[cite: 12]. 

**Root Directory:** `synthetic_xray_dataset`

* 📁 **Cardiac_Findings** (Cardiomegaly, vascular congestion)
* 📁 **Domain_Shift** (Scanner variations, resolution differences)
* 📁 **Imaging_Artifacts** (Motion blur, noise, exposure issues)
* 📁 **Infectious_Patterns** (Pneumonia, consolidation)
* 📁 **Lung_Opacities** (Ground-glass, diffuse haziness)
* 📁 **Medical_Devices** (Pacemakers, tubes, catheters)
* 📁 **Normal_Anatomy** (Healthy lungs, clear diaphragm)
* 📁 **Pleural_Conditions** (Effusion, pneumothorax)
* 📁 **Structural_Lesions** (Nodules, masses, fibrosis)
* 📁 **View_Positioning** (AP/PA views, rotation)

*(Structure verified from project file output)*

## 🛠️ Tech Stack & Dependencies
* **Language:** Python
* **Environment:** Google Colab (GPU-enabled)
* **Generative Model:** `runwayml/stable-diffusion-v1-5` (Hugging Face) [cite: 9]
* **Classification Model:** ResNet-18 (Pre-trained on ImageNet)
* **Libraries:**
    * `diffusers`, `transformers`, `accelerate` (For Generation)
    * `torch`, `torchvision` (For Training)
    * `matplotlib` (For Visualization)

## 🚀 How to Run

### Step 1: Install Dependencies
Run this command in the first cell of your notebook:
```bash
!pip install diffusers transformers accelerate torch torchvision
