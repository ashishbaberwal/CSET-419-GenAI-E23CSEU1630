# Lab 6 – Pix2Pix GAN (Image-to-Image Translation)

## Objective

Implement a Pix2Pix Generative Adversarial Network (GAN) using a **U-Net Generator** and **PatchGAN Discriminator** to perform paired image-to-image translation and compare the results with a baseline CNN encoder–decoder model.

---

## Workflow

### 1. Dataset Preparation

* Use a **paired dataset** where each input image corresponds to a target image.
* Example dataset: **Edges2Shoes**.
* Steps:

  * Load paired images (Edge image → Real image).
  * Normalize images to the range **[-1, 1]**.
  * Split dataset into **training** and **testing** sets.

---

### 2. Implement the Generator (U-Net)

* Build a **U-Net architecture** for the generator.
* Components:

  * Downsampling blocks using convolution layers.
  * Upsampling blocks using transposed convolutions.
  * Skip connections between encoder and decoder layers.
* Purpose:

  * Preserve spatial information.
  * Reduce information loss during encoding.

---

### 3. Implement the Discriminator (PatchGAN)

* Build a **PatchGAN discriminator** that evaluates image patches instead of the entire image.
* Architecture features:

  * Convolutional layers only.
  * No fully connected layers.
  * Outputs a **patch-level authenticity map** indicating real or fake regions.

---

### 4. Define Loss Functions

Two types of loss are used:

**Adversarial Loss**

* Encourages the generator to produce realistic images.

**L1 Reconstruction Loss**

* Ensures the generated image is close to the ground truth image.

**Total Generator Loss**
Generator Loss = Adversarial Loss + λ × L1 Loss

---

### 5. Model Training

* Use **Adam optimizer**.
* Learning rate: **0.0002**.

Training steps:

1. Pass input image to the generator to create a fake image.
2. Train discriminator using:

   * Real image pairs.
   * Generated image pairs.
3. Train generator using adversarial + reconstruction loss.
4. Alternate training between generator and discriminator.
5. Repeat for multiple epochs.

---

### 6. Generate Translated Images

* After training:

  * Input edge images to the generator.
  * Produce realistic output images.

---

### 7. Performance Comparison

Compare the outputs of:

* **Baseline CNN Encoder–Decoder (Lab 5)**
* **Pix2Pix GAN**

Observation:

* Pix2Pix produces **sharper and more realistic images** due to adversarial training.

---

## Expected Output

* Generated images with improved texture and sharpness.
* Visual comparison between CNN-based translation and Pix2Pix GAN results.

---

## Key Concepts Learned

* Conditional Generative Adversarial Networks (cGAN)
* U-Net architecture
* PatchGAN discriminator
* Adversarial training
* Image-to-image translation using GANs

---

## Tools and Libraries

* Python
* PyTorch / TensorFlow
* NumPy
* Matplotlib
* OpenCV

---

## Conclusion

Pix2Pix GAN significantly improves image translation quality compared to traditional encoder–decoder CNN models by using adversarial training and patch-level discrimination.
