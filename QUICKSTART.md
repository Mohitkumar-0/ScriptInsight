# Quick Start Guide

## Step 1: Install Dependencies

Open PowerShell and navigate to the project directory:

```powershell
cd "c:\Users\iquan\OneDrive\Desktop\Aditya Mishra"
pip install -r requirements.txt
```

## Step 2: Verify Model File

Make sure `best_model.h5` exists in the project directory. This file is created during the Jupyter notebook training.

## Step 3: Run the Application

```powershell
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
 * WARNING in app.run()
```

## Step 4: Access the Web Interface

Open your web browser and go to:
```
http://localhost:5000
```

## Step 5: Upload a Handwriting Image

1. Drag and drop an image of handwriting onto the upload area
2. Or click "Browse Files" to select an image
3. Wait for the model to analyze the image
4. View the personality prediction results

## Supported Image Formats

- PNG
- JPG / JPEG
- GIF
- BMP
- TIFF

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the Flask server.

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `OSError: [Errno 48] Address already in use` | Port 5000 is in use. Edit port in `app.py` |
| `Model not loaded` error | Ensure `best_model.h5` is in the project directory |
| Image upload fails | Check file size (max 16MB) and format |
| Slow predictions | Normal on CPU. Inference takes ~10-30 seconds |

## File Structure Created

```
Aditya Mishra/
├── app.py                    ✓ Backend Flask server
├── requirements.txt          ✓ Python dependencies
├── README.md                 ✓ Full documentation
├── QUICKSTART.md            ✓ This file
├── best_model.h5            ✓ Trained model (from notebook)
├── templates/
│   └── index.html           ✓ Web frontendss
├── static/
│   ├── css/
│   │   └── style.css       ✓ Styling
│   └── js/
│       └── script.js       ✓ Frontend logic
└── uploads/                 (Created automatically)
```

## Key Features

✅ Drag-and-drop image upload
✅ Real-time prediction with confidence scores
✅ Beautiful modern UI with responsive design
✅ Support for multiple image formats
✅ REST API endpoints for integration

## API Usage Example

You can also test the API using curl or any HTTP client:

```bash
# Upload and predict
curl -X POST -F "file=@handwriting.jpg" http://localhost:5000/api/predict

# Get app info
curl http://localhost:5000/api/info
```

Expected response:
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

## Performance Tips

1. Use clear, high-contrast handwriting images
2. Ensure the entire handwriting sample is visible
3. Avoid rotated or severely skewed images
4. Image should be at least 200×200 pixels

## Development Server

The application runs in debug mode, which means:
- Automatic reload on code changes
- Detailed error messages
- Development server only (not for production)

For production deployment, consider using:
- Gunicorn
- Waitress
- uWSGI

## Next Steps

1. Try uploading different handwriting samples
2. Experiment with different image qualities
3. Check the probability distributions for insights
4. Review the `personality_prediction_resnet.ipynb` to understand the model training

Enjoy using the Handwriting Personality Detector! 🖊️
