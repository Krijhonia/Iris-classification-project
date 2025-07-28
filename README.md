# Universal Flower Prediction System with Morphological Analysis

A sophisticated machine learning system that can identify flower species and analyze their morphological characteristics using computer vision and advanced ML techniques.

## Features

- **Multi-Species Classification**: Can identify various flower species including Iris (setosa, versicolor, virginica), rose, sunflower, daisy, tulip, daffodil, lily, and orchid
- **Morphological Analysis**: Estimates key flower measurements:
  - Sepal length and width
  - Petal length and width
- **Advanced Image Processing**:
  - Feature extraction using pixel data
  - Color histogram analysis
  - Texture analysis using OpenCV
  - Edge detection
- **Multiple Classification Methods**:
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
  - Rule-based classification (fallback when no training data available)
- **Interactive Interface**:
  - Single image prediction
  - Batch processing capabilities
  - Visual results with matplotlib
  - Morphological diagram generation

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - PIL (Pillow)
  - scikit-learn
  - matplotlib
  - pandas
  - seaborn
  - OpenCV (cv2)

## Usage

1. **Single Image Prediction**:
   ```python
   from flower_predictor import FlowerPredictor
   
   predictor = FlowerPredictor()
   image_path = "path/to/your/flower/image.jpg"
   species, confidence, measurements, _ = predictor.predict_flower(image_path)
   ```

2. **Batch Processing**:
   ```python
   folder_path = "path/to/folder/with/images"
   results = batch_predict_flowers(predictor, folder_path)
   ```

3. **Interactive CLI**:
   ```bash
   python flower_predictor.py
   ```

## Project Structure

- `flower_predictor.py`: Main class and prediction logic
- `gui_flower_predictor.py`: GUI interface for the prediction system
- `Iris_dataset/`: Directory containing training and test images
  - `iris-setosa/`
  - `iris-versicolour/`
  - `iris-virginica/`

## Output

The system provides:
1. Predicted flower species
2. Confidence score
3. Estimated morphological measurements
4. Visual representation including:
   - Original image
   - Prediction results
   - Morphological diagram

## Features in Detail

### Image Processing
- Pixel-level feature extraction
- Color histogram analysis (16 bins per channel)
- Edge detection using Canny algorithm
- Texture analysis using grayscale statistics

### Machine Learning
- Support for both KNN and Random Forest classifiers
- Feature scaling using StandardScaler
- Principal Component Analysis (PCA) capability
- Cross-validation support

### Morphological Analysis
- Uses both trained models and baseline measurements
- Adds realistic variations to measurements
- Visual representation through custom diagrams

## License

This project is available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!
