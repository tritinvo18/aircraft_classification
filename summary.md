# Aircraft Classification - Optimization & Workflow Summary

## 1. Data Augmentation Experiments
*   **RandAugment & Random Erasing**: We initially tested advanced automated data augmentations (`RandAugment`) and information-dropping techniques (`RandomErasing`). 
*   **Reversion**: Because these advanced augmentations negatively impacted performance (likely due to the fine-grained nature of aircraft features or insufficient training epochs), we reverted the training transforms back to the original baseline: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `RandomRotation(15)`, and `ColorJitter`.

## 2. Training Optimization & Early Stopping
*   Instead of further tweaking data augmentations, we shifted focus to optimization.
*   **Early Stopping**: We implemented early stopping mechanisms across the project:
    *   *Development Notebooks* (`development_test.ipynb`): Halts training if validation accuracy does not improve for 5 consecutive epochs.
    *   *Final Retraining* (`main_report.ipynb`): Halts training if the training loss does not improve for 5 consecutive epochs (since it trains on the full `trainval` dataset without a validation split).

## 4. Hyperparameter Tuning: LR Finder vs. Optuna
*   **LR Finder**: A script was provided to perform a rapid, single-epoch learning rate sweep. It exponentially increases the LR per batch, allowing you to visually pick the optimal learning rate (the steepest downward slope in the loss curve). It is extremely fast and computationally cheap.
*   **Optuna**: We compared this to Optuna, a comprehensive Bayesian optimization framework. While Optuna is fully automated and can tune multiple hyperparameters simultaneously (batch size, weight decay, LR), it is highly compute-intensive and slow compared to the LR Finder.
*   *Conclusion*: Use the LR Finder for rapid prototyping and Optuna for final, comprehensive model tuning.

---

### 1. Dynamic Pathing & Colab Detection

Never use absolute paths (e.g., C:/Users/... or /home/jimmy/...). Always use relative paths (e.g., FGVCAircraft_Subset20/).

If you plan to use Google Colab, you have to mount your Google Drive to access your files. A great practice is to put a "setup cell" at the very top of your notebooks that detects if it's running in Colab and sets a base path accordingly:

```python
import os
import sys

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    # Point this to your project folder in Google Drive
    BASE_DIR = '/content/drive/MyDrive/aircraft_classification/'
else:
    # Local or standard Jupyter server
    BASE_DIR = './'

# Then prepend BASE_DIR to your dataset and weight paths
data_path = os.path.join(BASE_DIR, 'FGVCAircraft_Subset20')
```

### 2. Extract Logic to Python Modules (.py)

Notebooks get messy when they contain hundreds of lines of training loops and model architectures. Since you already have a utils.py file, you should move your train_epoch, eval_epoch, and dataset class definitions into it.

To ensure your Jupyter Notebook / Colab automatically detects changes you make to utils.py without needing to restart the kernel, put this at the top of your notebook:

```python
%load_ext autoreload
%autoreload 2
import utils
```

### 3. Automated Dataset Retrieval

If your dataset (FGVCAircraft_Subset20) is large, moving it to Colab manually can be painful. It's best practice to host the zipped dataset somewhere (like Google Drive, AWS S3, or Kaggle) and write a cell that downloads and unzips it if it isn't found locally:

```python
if not os.path.exists(data_path):
    print("Dataset not found. Downloading...")
    # e.g., !gdown --id <google_drive_file_id>
    # !unzip dataset.zip -d {BASE_DIR}
```

### 4. Hardware Agnosticism

Always detect the GPU dynamically so the notebook runs fine on a local CPU server or a Colab GPU instance. (It looks like you are already doing this, but it's a golden rule):

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 5. Environment Synchronization (requirements.txt)

You already have a requirements.txt. In Colab, default packages are often outdated or different from your local setup. It is good practice to include a cell at the top of the notebook that quietly installs your dependencies:

```python
# In a notebook cell:
!pip install -r {BASE_DIR}requirements.txt -q
```

### Summary of an Ideal Project Structure:

*   `src/` or `utils/`: Contains all your .py files (models, training loops, data loaders).
*   `notebooks/`: Contains your .ipynb files, which should strictly be used for running experiments, visualizations, and orchestrating the functions imported from src/.
*   `data/`: Added to .gitignore. Downloaded dynamically.
*   `weights/`: Saved model checkpoints. In Colab, ensure this points to your mounted Drive so you don't lose them if the session crashes.

---

```python
import math
import copy
import torch
import matplotlib.pyplot as plt

def find_learning_rate(model, train_loader, criterion, optimizer, device, init_value=1e-8, final_value=10.0, beta=0.98):
    """
    Automated Learning Rate Finder.

    Args:
        model: Your PyTorch model
        train_loader: DataLoader for the training set
        criterion: Loss function
        optimizer: Optimizer (e.g., optim.SGD or optim.AdamW)
        device: torch.device ('cuda' or 'cpu')
        init_value: Starting learning rate (very small)
        final_value: Ending learning rate (very large)
        beta: Smoothing factor for the loss
    """
    print("Starting LR Finder...")

    # 1. Save original states so we don't ruin the initialized model
    original_model_state = copy.deepcopy(model.state_dict())
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())

    # Calculate the exponential factor to multiply the LR by at each batch
    num_batches = len(train_loader)
    mult = (final_value / init_value) ** (1 / num_batches)

    # Set the initial learning rate
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []

    model.train()

    for inputs, labels in train_loader:
        batch_num += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            print("Loss exploded, stopping early.")
            break

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store values for plotting
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Backprop and update weights
        loss.backward()
        optimizer.step()

        # Update the learning rate for the next batch
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    # 2. Restore original model and optimizer states
    print("Restoring original model states...")
    model.load_state_dict(original_model_state)
    optimizer.load_state_dict(original_optimizer_state)

    # 3. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(log_lrs, losses)
    plt.xlabel("Learning Rate (Log10 Scale)")
    plt.ylabel("Smoothed Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    plt.show()

    print("LR Finder complete. Look for the steepest downward slope in the graph!")

# ==========================================
# HOW TO USE IT IN YOUR NOTEBOOK:
# ==========================================
# 1. Initialize your model, criterion, and optimizer as usual
# model = MyModel().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-8) # The LR here will be overwritten
#
# 2. Call the function
# find_learning_rate(model, train_loader, criterion, optimizer, device)
```

### How to interpret the plot:

When you run this, a graph will appear.

1. Don't pick the learning rate where the loss is the lowest. At the absolute bottom of the curve, the learning rate is actually too high and the model is on the verge of diverging.
2. Pick the learning rate where the curve is steepest going downwards. Usually, this is about one order of magnitude (10x smaller) before the absolute minimum loss is reached.
3. For example, if the lowest loss happens at 10^-2 (0.01), a great starting learning rate is usually around 10^-3 (0.001).

---

### 1. Automated Augmentation Strategies

Instead of manually tuning augmentations, you can use policies discovered through reinforcement learning that apply a series of transformations:

*   [RandAugment](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html): Extremely effective and easy to use. It randomly selects a set number of augmentations and applies them with a given magnitude.
    ```python
    transforms.RandAugment(num_ops=2, magnitude=9)
    ```

*   [TrivialAugmentWide](https://pytorch.org/vision/stable/generated/torchvision.transforms.TrivialAugmentWide.html): A simpler, parameter-free alternative to RandAugment that often works just as well or better.
    ```python
    transforms.TrivialAugmentWide()
    ```

### 2. Information Dropping Techniques

These force the model to learn features from multiple parts of the aircraft (like wings, tail, engines) rather than relying on a single distinguishing feature:

*   [Random Erasing / Cutout](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html): Randomly selects a rectangle region in an image and replaces its pixels with random values or the mean pixel value. Note: this is typically applied after converting the image to a tensor.
    ```python
    # After transforms.ToTensor()
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ```

### 3. Mixing Images (Advanced)

These are applied at the batch level during the training loop rather than in the Dataset transforms, but they are highly effective for fine-grained classification tasks:

*   **Mixup**: Takes two random images and linearly interpolates their pixels and their labels. This smooths the decision boundary.
*   **CutMix**: Takes a patch from one image and pastes it onto another, mixing their labels proportionally to the area of the patch. It acts as both regularization and localization training.

### 4. Additional Geometric and Photometric Transforms

*   [RandomAffine](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html): Can introduce shear and translation which might mimic different camera angles capturing the aircraft.
    ```python
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
    ```

*   [RandomPerspective](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomPerspective.html): Simulates viewing the aircraft from different 3D angles.
    ```python
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
    ```

*   [GaussianBlur](https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html): Simulates out-of-focus cameras or motion blur.
    ```python
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    ```

### Recommendation for Aircraft Classification:
Aircraft classification is a "fine-grained" visual categorization task (species/models look very similar). Techniques like Cutout/Random Erasing and CutMix are usually exceptionally helpful because they force the neural network to look at the entire aircraft (e.g., identifying by the tail design if the nose is erased) rather than relying on one specific feature.
