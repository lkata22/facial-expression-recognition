# Facial Expression Recognition

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge


პროექტი შესრულებულია google colab-ში და tracking-ისთვის გამოყენებულია wandb.

---

## Repository Structure

src/
data_loader.py
model.py
train.py
utils.py
notebooks/
facial_expression_pipeline.ipynb
requirements.txt
README.md


---

## გამოყენებული მოდელები და შედეგები

### SimpleCNN
- Epochs: 15
- Learning Rate: 1e-3
- Batch Size: 64
- **Validation Accuracy:** ~53%

### DeeperCNN
- Epochs: 25
- Learning Rate: 1e-3
- Batch Size: 64
- **Validation Accuracy:** ~59%


###  DeeperCNN2 (5 conv layers)
- Epochs: 25
- Learning Rate: 1e-3
- Batch Size: 64
- **Validation Accuracy:** ~61-63%

###  ResNet18 (Transfer Learning)
- Epochs: 30
- Learning Rate: 1e-4
- Batch Size: 64
- **Validation Accuracy:** ~50% +


---

## Experiment Tracking 

### WandB Project: [facial_expression_recognition](https://wandb.ai/lkata22-free-university-of-tbilisi-/facial_expression_recognition)

- [SimpleCNN Run Link](https://wandb.ai/lkata22-free-university-of-tbilisi-/facial_expression_recognition/runs/lnydryyf?nw=nwuserlkata22)
- [DeeperCNN Run Link](https://wandb.ai/lkata22-free-university-of-tbilisi-/facial_expression_recognition/runs/lsdqx8sn?nw=nwuserlkata22)
- [DeeperCNN2 Run Link](https://wandb.ai/lkata22-free-university-of-tbilisi-/facial_expression_recognition/runs/2k0vgoy6?nw=nwuserlkata22)
- [ResNet18 Run Link](https://wandb.ai/lkata22-free-university-of-tbilisi-/facial_expression_recognition/runs/6mstew2k?nw=nwuserlkata22)


