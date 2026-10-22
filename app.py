import os
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Always revalidate static JS/CSS so fixes are not hidden by stale caches

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Personality classes
PERSONALITY_CLASSES = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff'}
MODEL_PATH = 'best_model.h5'

model = None


def build_resnet_model(keras):
    """Rebuild the notebook's ResNet50 + dense head architecture (same layer names as the saved model)."""
    base_model = keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    return keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d'),
        keras.layers.Dense(512, activation='relu', name='dense_1'),
        keras.layers.BatchNormalization(name='batch_normalization'),
        keras.layers.Dropout(0.5, name='dropout'),
        keras.layers.Dense(256, activation='relu', name='dense_2'),
        keras.layers.BatchNormalization(name='batch_normalization_1'),
        keras.layers.Dropout(0.4, name='dropout_1'),
        keras.layers.Dense(128, activation='relu', name='dense_3'),
        keras.layers.BatchNormalization(name='batch_normalization_2'),
        keras.layers.Dropout(0.3, name='dropout_2'),
        keras.layers.Dense(len(PERSONALITY_CLASSES), activation='softmax', name='output')
    ])


def load_keras3_h5_weights(target_model, path):
    """Load weights from an H5 file saved by Keras 3, which TF 2.13's load_model cannot deserialize."""
    import h5py

    def collect(group):
        weights = {}
        group.visititems(lambda name, obj: weights.__setitem__(name.split('/')[-1], obj[()])
                         if isinstance(obj, h5py.Dataset) else None)
        return weights

    with h5py.File(path, 'r') as f:
        saved = f['model_weights']
        base_model = target_model.get_layer('resnet50')
        layers = [(layer, saved[layer.name]) for layer in target_model.layers
                  if layer.weights and layer.name != 'resnet50']
        layers += [(layer, saved['resnet50'][layer.name]) for layer in base_model.layers if layer.weights]
        for layer, group in layers:
            stored = collect(group)
            layer.set_weights([stored[w.name.split('/')[-1].split(':')[0]] for w in layer.weights])


try:
    import tensorflow as tf
    from tensorflow import keras
    # tensorflow-metal returns corrupted (saturated) outputs from model.predict for this model on the
    # Apple GPU; CPU inference is correct and fast enough for single images.
    tf.config.set_visible_devices([], 'GPU')
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print('✓ Model loaded successfully!')
    except Exception as e:
        print(f'⚠ load_model failed ({e.__class__.__name__}); rebuilding architecture and loading weights...')
        try:
            model = build_resnet_model(keras)
            load_keras3_h5_weights(model, MODEL_PATH)
            print('✓ Trained weights loaded into rebuilt ResNet50 model!')
        except Exception as e:
            print(f'✗ Could not load trained model weights: {e}')
            model = None
except Exception as e:
    print(f'✗ TensorFlow import failed: {e}')


def preprocess_image(image_path):
    """Preprocess image exactly like the training notebook: grayscale, 224x224, /255, stacked to 3 channels."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # OpenCV's imread has no GIF decoder; read the first frame via VideoCapture instead.
            capture = cv2.VideoCapture(image_path)
            ok, frame = capture.read()
            capture.release()
            if ok and frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img is None:
            return None, 'Could not read the image. The file may be corrupted or in an unsupported format.'

        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.stack([img] * 3, axis=-1)
        img_batch = np.expand_dims(img, axis=0)
        return img_batch, None
    except Exception as e:
        return None, str(e)


def predict_personality(image_path):
    """Predict personality from handwriting image."""
    img_batch, error = preprocess_image(image_path)
    if error:
        return None, error
    if img_batch is None:
        return None, 'Failed to preprocess image'

    try:
        prediction = model.predict(img_batch, verbose=0)[0]
        predicted_class_idx = int(np.argmax(prediction))
        predicted_class = PERSONALITY_CLASSES[predicted_class_idx]
        confidence = float(prediction[predicted_class_idx])
        result = {
            'predicted_personality': predicted_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                PERSONALITY_CLASSES[i]: round(float(prediction[i]), 4)
                for i in range(len(PERSONALITY_CLASSES))
            }
        }
        return result, None
    except Exception as e:
        print(f'⚠ Prediction failed: {e}')
        return None, f'Model prediction failed: {e}'


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for personality prediction"""
    try:
        if model is None:
            return jsonify({'error': 'The personality model is not loaded on the server. Check the server logs.'}), 503

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400

        # Save file temporarily under a unique, sanitized name
        filename = f'{uuid.uuid4().hex}_{secure_filename(file.filename) or "upload." + extension}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Make prediction
            result, error = predict_personality(filepath)
            if error:
                return jsonify({'error': error}), 422
            return jsonify({'success': True, 'prediction': result}), 200
        finally:
            # Clean up - delete the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    except HTTPException:
        raise  # e.g. 413 from MAX_CONTENT_LENGTH, handled by its errorhandler below
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.errorhandler(413)
def file_too_large(_error):
    """Return JSON (not Flask's HTML page) when the upload exceeds MAX_CONTENT_LENGTH"""
    return jsonify({'error': 'Image is too large. Please select an image smaller than 16MB.'}), 413


@app.route('/api/info', methods=['GET'])
def api_info():
    """Get model and app information"""
    return jsonify({
        'app_name': 'Handwriting-Based Personality Detector',
        'model': 'ResNet50 with Dense Layers',
        'personality_classes': PERSONALITY_CLASSES,
        'model_loaded': model is not None
    }), 200


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5001)
