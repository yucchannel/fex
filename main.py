import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

# Load and resize image, convert to grayscale
def load_image(path, target_size=(128, 128)):
    try:
        img = Image.open(path).convert('L')
        img = img.resize(target_size)
        return np.array(img)
    except Exception as e:
        print(f"Failed to load image: {path}, Error: {e}")
        return None

# Apply Sobel filter for edge detection
def sobel_filter(image):
    S_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    S_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    height, width = image.shape
    G_x = np.zeros_like(image, dtype=np.float32)
    G_y = np.zeros_like(image, dtype=np.float32)
    G = np.zeros_like(image, dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            region = image[y - 1:y + 2, x - 1:x + 2]
            G_x[y, x] = np.sum(region * S_x)
            G_y[y, x] = np.sum(region * S_y)
            G[y, x] = np.sqrt(G_x[y, x]**2 + G_y[y, x]**2)

    return G

# Flatten edge image to feature vector
def extract_features(image_path, target_size=(128, 128)):
    img = load_image(image_path, target_size)
    if img is None:
        return None
    edge_img = sobel_filter(img)
    return edge_img.flatten()

# Euclidean distance
def calculate_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# Load dataset and extract features
def prepare_data(dataset_dir, target_size=(128, 128)):
    X, y = [], []

    # Positive samples (red signal)
    red_dir = os.path.join(dataset_dir, "reds")
    for img_file in os.listdir(red_dir):
        if img_file.startswith('.') or not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(red_dir, img_file)
        features = extract_features(img_path, target_size)
        if features is not None:
            X.append(features)
            y.append(1)

    # Negative samples (non-red signal)
    other_dir = os.path.join(dataset_dir, "other")
    for img_file in os.listdir(other_dir):
        if img_file.startswith('.') or not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(other_dir, img_file)
        features = extract_features(img_path, target_size)
        if features is not None:
            X.append(features)
            y.append(0)

    return np.array(X), np.array(y)

# Train the classifier using average vectors
def prepare_and_train(dataset_dir, target_size=(128, 128)):
    X, y = prepare_data(dataset_dir, target_size)
    cat_features = X[y == 1]
    non_cat_features = X[y == 0]
    cat_avg = np.mean(cat_features, axis=0)
    non_cat_avg = np.mean(non_cat_features, axis=0)
    return cat_avg, non_cat_avg

# Predict using distance to average vectors
def predict_image(cat_avg, non_cat_avg, image_path, target_size=(128, 128)):
    features = extract_features(image_path, target_size)
    if features is None:
        return "Image loading failed"
    
    cat_distance = calculate_distance(features, cat_avg)
    non_cat_distance = calculate_distance(features, non_cat_avg)

    if cat_distance < non_cat_distance:
        return "Red signal"
    else:
        return "Not red signal"

# Read input.txt and write output.json
def main():
    dataset_dir = 'dataset'
    input_file = 'input.txt'
    output_file = 'output.json'
    cat_avg, non_cat_avg = prepare_and_train(dataset_dir)

    results = []

    # Read input paths from file
    with open(input_file, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    for image_path in image_paths:
        prediction = predict_image(cat_avg, non_cat_avg, image_path)
        print(f"{image_path} -> {prediction}")
        results.append({"image": image_path, "prediction": prediction})

    # Write output to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
