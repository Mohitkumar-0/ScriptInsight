# Handwriting-Based Personality Detection Web Application

A web application that predicts personality traits from handwritten images using a ResNet50 deep learning model.

## Features

- 🖼️ **Image Upload**: Drag and drop or browse to upload handwriting images
- 🧠 **AI Prediction**: Uses ResNet50 model to analyze handwriting patterns
- 📊 **Personality Traits**: Predicts Big Five personality traits:
  - Agreeableness
  - Conscientiousness
  - Extraversion
  - Neuroticism
  - Openness
- 📈 **Confidence Scores**: Shows probability distribution across all traits
- 🎨 **Modern UI**: Beautiful, responsive web interface

## Project Structure

```
Aditya Mishra/
├── app.py                          # Flask backend application
├── requirements.txt                # Python dependencies
├── best_model.h5                   # Trained ResNet50 model
├── personality_prediction_resnet.ipynb  # Training notebook
├── templates/
│   └── index.html                  # Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css              # Frontend styling
│   └── js/
│       └── script.js              # Frontend JavaScript
└── uploads/                        # Temporary upload folder (auto-created)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\iquan\OneDrive\Desktop\Aditya Mishra"
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file exists**:
   - Ensure `best_model.h5` is in the project root directory
   - This file should have been generated during training

## Running the Application

### Start the Flask server:

```bash
python app.py
```

You should see output like:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Access the web interface:

1. Open your browser
2. Navigate to: `http://localhost:5000`
3. You should see the Handwriting Personality Detector interface

## Usage

1. **Upload an Image**:
   - Drag and drop a handwriting image into the upload area, OR
   - Click "Browse Files" to select an image from your computer

2. **Supported Formats**:
   - PNG, JPG, JPEG, GIF, BMP, TIFF
   - Maximum file size: 16MB

3. **View Results**:
   - The model will analyze the image
   - Primary personality trait will be displayed prominently
   - A probability distribution chart shows confidence for each trait

4. **Upload Another Image**:
   - Click "Upload Another Image" to make a new prediction

## API Endpoints

### POST `/api/predict`
Uploads an image and returns personality predictions.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**:
```json
{
  "success": true,
  "prediction": {
    "predicted_personality": "Conscientiousness",
    "confidence": 0.5115,
    "probabilities": {
      "Agreeableness": 0.0915,
      "Conscientiousness": 0.4250,
      "Extraversion": 0.1425,
      "Neuroticism": 0.3135,
      "Openness": 0.0275
    }
  }
}
```

### GET `/api/info`
Returns application and model information.

**Response**:
```json
{
  "app_name": "Handwriting-Based Personality Detector",
  "model": "ResNet50 with Dense Layers",
  "personality_classes": ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"],
  "model_loaded": true
}
```

## Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Layers**: 3 Dense layers (512 → 256 → 128 neurons)
- **Total Parameters**: 24.8M
- **Input Shape**: 224×224×3 (RGB images)
- **Output**: 5 classes (personality traits)

## Model Performance

| Personality | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Conscientiousness | 51.15% | 0.27 | 0.51 | 0.35 |
| Openness | 40.28% | 0.33 | 0.40 | 0.36 |
| Agreeableness | 23.36% | 0.48 | 0.23 | 0.31 |
| Extraversion | 17.86% | 0.29 | 0.18 | 0.22 |
| Neuroticism | 16.91% | 0.23 | 0.17 | 0.19 |

**Overall Test Accuracy**: 30.30%

## Training Details

- **Dataset**: 3,416 training images, 660 test images
- **Classes**: 5 Big Five personality traits
- **Optimizer**: Adam (initial lr=0.001, fine-tuning lr=1e-5)
- **Loss Function**: Categorical Crossentropy
- **Training Strategy**: Transfer Learning + Fine-tuning
- **Early Stopping**: Enabled (patience=10)
- **Learning Rate Scheduling**: Enabled (factor=0.5, patience=5)

## Troubleshooting

### Model not loading
- **Error**: "Model not loaded"
- **Solution**: Ensure `best_model.h5` is in the project root directory

### Port already in use
- **Error**: `OSError: [Errno 48] Address already in use`
- **Solution**: Change the port in `app.py` (line: `app.run(port=5000)`)

### Import errors
- **Error**: `ModuleNotFoundError`
- **Solution**: Ensure all requirements are installed:
  ```bash
  pip install -r requirements.txt
  ```

### Large image processing
- If image processing is slow:
  - Try with a smaller resolution image
  - Check your system's available memory

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Model**: ResNet50 with custom dense layers

## Performance Notes

- Image preprocessing time: ~1-2 seconds
- Model inference time: ~10-30 seconds (depends on system)
- CPU processing: Model runs on CPU (GPU support available with CUDA)

## Future Enhancements

- [ ] Add GPU support for faster inference
- [ ] Support for batch image processing
- [ ] Historical prediction tracking
- [ ] Export results as PDF
- [ ] Mobile app version
- [ ] Real-time camera capture
- [ ] Advanced handwriting analysis features

## Limitations

1. Model trained on specific dataset of handwriting samples
2. Accuracy varies based on image quality
3. Works best with clear handwriting samples
4. Requires proper image preprocessing

## License

This project is for educational purposes.

## Contact & Support

For issues or questions, please refer to the training notebook: `personality_prediction_resnet.ipynb`

## Disclaimer

This application provides predictions based on machine learning models and should not be used as a substitute for professional psychological evaluation or diagnosis.
