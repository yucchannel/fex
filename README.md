# FEX - Feature Extraction System

FEX (Feature Extraction System) is an object recognition system that utilizes image processing techniques—such as Sobel edge detection—to analyze and classify objects based on their unique features. Built with Python, FEX is fully customizable and can be trained with your own dataset to recognize a wide variety of objects.

## Features

- **Grayscale Conversion**: Converts input images to grayscale for consistent preprocessing.
- **Sobel Edge Detection**: Detects edges in images, helping to extract meaningful object features.
- **Customizable Dataset**: Easily train the system on your own images by organizing them into class-specific folders.
- **Real-Time Processing**: Supports processing of images or video streams in real-time, ideal for surveillance or live monitoring applications.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/yucchannel/fex.git
```
### Step 2: Install dependencies

```bash
cd FEX
pip install -r requirements.txt
```

### Step 3: Prepare your dataset

Place your dataset into the dataset/ folder. Create subfolders for each class (e.g., cat/, dog/, etc.), and place the corresponding images inside them.

### Step 4: Run the system

```bash
python main.py
```

### Contact

If you have any questions, suggestions, or feedback, feel free to reach out via GitHub Issues or email:
yucchanneltv@gmail.com
