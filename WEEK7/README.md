# Lab 7 – Neural Style Transfer (NST)

## Objective

To implement **Neural Style Transfer (NST)** using a pretrained CNN and generate a stylized image by combining:

* **Content** from one image
* **Style** from another image

The experiment uses a pretrained **VGG19 network** to extract content and style features and then optimizes an image to blend both characteristics.

---

## Workflow

### 1. Data Preparation

1. Load the **content image** (the image whose structure you want to keep).
2. Load the **style image** (the image whose artistic style will be applied).
3. Resize images to the same dimensions.
4. Convert images into tensors and normalize them according to the pretrained model requirements.

---

### 2. Load Pretrained CNN

1. Load a pretrained **VGG19 model**.
2. Freeze all network weights so they are **not updated during training**.
3. Use intermediate layers of the network to extract features.

Purpose:

* Early layers capture **texture and colors**.
* Deeper layers capture **high-level content structure**.

---

### 3. Extract Feature Representations

Select layers for:

**Content Representation**

* A deeper CNN layer that captures object structure.

**Style Representation**

* Multiple shallow and intermediate layers to capture textures and patterns.

---

### 4. Define Loss Functions

#### Content Loss

Measures the difference between:

* Content features of the generated image
* Content features of the original content image

#### Style Loss

Measures the difference between style representations using **Gram matrices** of feature maps.

#### Total Loss

Total Loss = Content Loss + Style Loss

A weighting factor is used to control the balance between style and content.

---

### 5. Initialize Generated Image

* Start with:

  * the **content image**, or
  * **random noise**.

This image will be **optimized iteratively**.

---

### 6. Optimization Process

1. Pass the generated image through the VGG19 network.
2. Compute content loss and style loss.
3. Calculate total loss.
4. Use gradient descent to update the generated image.
5. Repeat for many iterations until the image converges.

---

### 7. Generate Stylized Image

After optimization:

* The final generated image should:

  * preserve the **content structure**
  * adopt the **style texture and artistic patterns**.

---

## Expected Output

* Stylized image where:

  * content from the content image is preserved
  * artistic style from the style image is transferred. 

---

## Key Concepts Learned

* Neural Style Transfer
* Feature extraction using pretrained CNNs
* Content and style representation
* Gram matrix for style representation
* Optimization-based image generation

---

## Tools and Libraries

* Python
* PyTorch / TensorFlow
* NumPy
* Matplotlib
* PIL / OpenCV

---

## Conclusion

Neural Style Transfer demonstrates how deep neural networks can separate **content and style information** from images and recombine them to produce visually artistic results.
