Here’s your **README.md** properly formatted in Markdown:

```markdown
# FEX - Feature Extraction System

FEX is an object recognition system that uses image processing techniques, including Sobel edge detection, to analyze and classify objects based on their unique features. Built with Python, it can be easily trained to recognize any object by using a custom dataset.

## Features

- **Grayscale Conversion**: Converts input images to grayscale to ensure uniformity during processing.
- **Sobel Edge Detection**: Detects edges in images, which are key for recognizing objects.
- **Customizable**: Train the system to recognize any object by adding a suitable dataset.
- **Real-Time Processing**: Capable of processing images or video streams in real-time, ideal for surveillance or live monitoring applications.

## Installation

### Step 1: Clone the repository
Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/FEX.git
```

### Step 2: Install the required dependencies
Navigate to the project directory and install the dependencies using `pip`:

```bash
cd FEX
pip install -r requirements.txt
```

### Step 3: Prepare your dataset
Place your dataset of labeled images inside the `dataset/` folder. Organize the dataset into subfolders representing different classes (e.g., `cat/`, `dog/`, etc.).

### Step 4: Run the system
Start the system by running:

```bash
python main.py
```

## Usage

- **Training the system**: 
  Place your labeled images in the `dataset/` folder. Make sure the images are organized into class-specific subfolders (e.g., `cats/`, `non_cats/`).

- **Prediction**: 
  The system will automatically compute the average feature vectors for each class and perform object recognition in real-time. Simply provide the path to the image you want to classify.

## Example Output

Here’s an example of the output you’ll see during prediction:

```
Prediction for image: new_image.jpg
Distance to cat: 5.6
Distance to non-cat: 8.2
Predicted class: Cat
```

## Contributing

Feel free to contribute to this project by submitting issues, suggestions, or pull requests. Contributions are encouraged, and we welcome new features and improvements!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For more information or to report any issues, feel free to reach out via GitHub Issues or email at [yucchanneltv@gmail.com].
```

### Key Sections in the Markdown:

- **Introduction**: A brief overview of what the project does.
- **Features**: Highlights of what the system can do.
- **Installation**: Instructions for setting up the project on your machine.
- **Usage**: How to train the model and use it for prediction.
- **Example Output**: Sample output of a prediction result.
- **Contributing**: Encouraging open-source contributions.
- **License**: Information about the project's open-source license.
- **Contact**: How users can contact you for support or inquiries.

