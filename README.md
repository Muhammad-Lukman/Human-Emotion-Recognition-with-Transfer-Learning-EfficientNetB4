# Human Emotion Recognition with Transfer Learning (EfficientNetB4)

This project implements **human emotion recognition** using deep learning on the [Human Emotions Dataset (Kaggle)](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes).
We benchmark a **baseline CNN (LeNet)** (upload before) against a **transfer learning approach (EfficientNetB4)**.

We implemented Transfer Learning this time and used a pretrained model with CovNet Layers freezed, and added a customized classification head and initialized its weights from scratch.

---

## Dataset

* **Name:** Human Emotions Dataset
* **Source:** [Kaggle – muhammadhananasghar/human-emotions-datasethes](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)
* **Classes:** `Angry`, `Happy`, `Sad`
* **Format:** Images organized in subfolders

Dataset download:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadhananasghar/human-emotions-datasethes")
print("Path to dataset files:", path)
```

---

## Models

### 1. Baseline: **LeNet (Custom CNN)**

* Val Accuracy: 78.7%
* Val Loss: 0.553 
* Angry Recall: 71% 
* Sad Recall: 77%
* Happy Recall: ~82–84%

The model was **biased toward the Happy class**. With class weighting and LR scheduling, recall improved, but overall generalization was limited. This showed the need for transfer learning with deeper architectures.

---

### 2. Transfer Learning: **EfficientNetB4**

* Pretrained weights (`imagenet`) + fine-tuned on dataset.
* Significant improvement was seen across accuracy and generaliziability of model.

**Validation Results (EfficientNetB4):**

* Validation Accuracy: 81.7% (vs. 78.7% LeNet)
* Validation Loss: 0.48 (lower, better generalization)
  
* Model became **balanced** across all three emotions unlike the previous LeNet model.

---

## Saved Models

Two versions of EfficientNetB4 were saved:

* Functional API version (Better Generaliziability):
  `Func_EmotionDetectorModelSaved.keras`
* Sequential API version (best weights):
  `EfficientNetEmotionModelBest.keras`

---

## Usage

### 1. Load Saved Models

```python
from tensorflow import keras

# Functional API Model
func_model = keras.models.load_model("./models/Func_EmotionDetectorModelSaved.keras")

# Sequential Model (Best)
seq_model = keras.models.load_model("./models/EfficientNetEmotionModelBest.keras")
```

---

### 2. Evaluate Model

```python
# Evaluate on validation dataset
val_loss, val_acc = seq_model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2f}")
print(f"Validation Loss: {val_loss:.2f}")
```

---

### 3. Make Predictions

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess new image
img = image.load_img("test_image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict with Sequential model
predictions = seq_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted Class:", predicted_class)
```

---

## Future Work

* Fine-tuning deeper EfficientNet layers for higher accuracy.
* Visualizing intermediate layers to interpret model decisions.
* Applying **Grad-CAM** to understand which image regions influence predictions most.

---

## Reference

* Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. [Paper Link](https://arxiv.org/abs/1905.11946)

---

With transfer learning, the model improved from a biased baseline CNN to a **balanced, production-quality classifier** suitable for further fine-tuning and research.
