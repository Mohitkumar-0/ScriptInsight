"""ScriptInsight Flask application.

Run:  python app.py            (serves http://127.0.0.1:5001 by default; see config.py)
"""
import logging
import platform
import time
from importlib.metadata import version as package_version
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException

import config
from src.errors import ModelUnavailableError
from src.preprocessing.image_processor import ImageValidationError, decode_image
from src.services.prediction_service import PredictionService, load_json

log = logging.getLogger('scriptinsight')


def error(message, status, code):
    return jsonify({'success': False, 'error': message, 'code': code}), status


def _round_scores(entry):
    entry = dict(entry)
    if 'scores' in entry:
        entry['scores'] = {k: round(v, 4) for k, v in entry['scores'].items()}
    if 'confidence' in entry:
        entry['confidence'] = round(entry['confidence'], 4)
    return entry


def prediction_payload(primary):
    """Primary-model result. Multiclass keeps the original response keys for backward compatibility."""
    base = {'model': primary['model'], 'display_name': primary['display_name'], 'task': primary['task'],
            'score_type': primary['score_type']}
    if primary['task'] == 'multiclass':
        return {**base, 'predicted_personality': primary['predicted_label'],
                'confidence': round(primary['confidence'], 4),
                'probabilities': {k: round(v, 4) for k, v in primary['scores'].items()}}
    return {**base, 'traits': {t: {'probability': round(primary['scores'][t], 4), 'level': primary['levels'][t]}
                               for t in config.TRAITS if t in primary['scores']}}


def create_app(service=None):
    app = Flask(__name__)
    # Room for multipart overhead on top of the image limit; decode_image enforces the exact limit.
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES + 64 * 1024
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.extensions['prediction_service'] = service if service is not None else PredictionService()
    started_at = time.time()

    def svc():
        return app.extensions['prediction_service']

    @app.context_processor
    def asset_helpers():
        def asset_url(filename):
            # Version stamp from the file's mtime so browsers never run a stale script or stylesheet.
            path = Path(app.static_folder) / filename
            version = int(path.stat().st_mtime) if path.exists() else 0
            return url_for('static', filename=filename, v=version)
        return {'asset_url': asset_url, 'disclaimer': config.DISCLAIMER, 'app_version': config.APP_VERSION}

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/api/predict', methods=['POST'])
    @app.route('/predict', methods=['POST'])
    def api_predict():
        upload = request.files.get('file')
        if upload is None:
            return error('No file provided. Send the image as multipart/form-data field "file".', 400, 'no_file')
        if not upload.filename:
            return error('No file selected.', 400, 'no_file')
        try:
            # The image is decoded in memory and never written to disk.
            decoded = decode_image(upload.read(config.MAX_UPLOAD_BYTES + 1), upload.filename)
            primary, results, warnings = svc().predict(decoded.gray)
        except ImageValidationError as exc:
            return error(exc.message, exc.http_status, exc.code)
        except ModelUnavailableError:
            return error('No prediction model is available on the server. Check /api/health.', 503, 'model_unavailable')
        except Exception:
            log.exception('Prediction failed')
            return error('The image could not be analysed because of a server error.', 500, 'server_error')

        return jsonify({
            'success': True,
            'prediction': prediction_payload(primary),
            'model_predictions': [_round_scores(r) for r in results],
            'image': {'width': decoded.width, 'height': decoded.height, 'format': decoded.format},
            'model_metadata': {'primary_model': primary['model'], 'trait_order': config.TRAITS,
                               'app_version': config.APP_VERSION},
            'warnings': warnings,
            'disclaimer': config.DISCLAIMER,
        }), 200

    @app.route('/api/info', methods=['GET'])
    def api_info():
        return jsonify({
            'app_name': config.APP_NAME,
            'app_version': config.APP_VERSION,
            'traits': config.TRAITS,
            'primary_model': svc().primary,
            'models': svc().model_info(),
            'upload': {'max_bytes': config.MAX_UPLOAD_BYTES, 'min_side_px': config.MIN_IMAGE_SIDE,
                       'extensions': sorted(config.ALLOWED_EXTENSIONS), 'field': 'file'},
            'disclaimer': config.DISCLAIMER,
        }), 200

    @app.route('/api/health', methods=['GET'])
    def api_health():
        available = [name for name, s in svc().status.items() if s.get('available')]
        healthy = bool(available)
        return jsonify({
            'status': 'ok' if healthy else 'degraded',
            'models_available': available,
            'models_unavailable': {n: s.get('state') for n, s in svc().status.items() if not s.get('available')},
            'uptime_seconds': round(time.time() - started_at, 1),
            'versions': {'python': platform.python_version(), 'flask': package_version('flask'), 'app': config.APP_VERSION},
        }), 200 if healthy else 503

    @app.route('/api/metrics', methods=['GET'])
    def api_metrics():
        evaluation = load_json(config.EVALUATION_DIR / 'latest.json')
        legacy = load_json(config.LEGACY_RECORDS_PATH)
        resnet18_history = load_json(config.RESNET18_HISTORY_PATH)
        return jsonify({
            'evaluation_available': evaluation is not None,
            'evaluation': evaluation,
            'plots': [url_for('evaluation_artifact', filename=p) for p in (evaluation or {}).get('plots', [])],
            'dataset_split': load_json(config.SPLITS_DIR / 'split_meta.json'),
            'training_history': {'resnet18': resnet18_history,
                                 'legacy_resnet50': {'epochs': legacy['history'], 'note': legacy['history_note'],
                                                     'provenance': legacy['provenance']} if legacy else None},
            'legacy_recorded': {k: v for k, v in legacy.items() if k != 'history'} if legacy else None,
            'accuracy_target': {'target': config.ACCURACY_TARGET, 'traits': config.ACCURACY_TARGET_TRAITS},
        }), 200

    @app.route('/artifacts/evaluation/<path:filename>')
    def evaluation_artifact(filename):
        if Path(filename).suffix.lower() not in {'.png', '.json'}:
            abort(404)
        return send_from_directory(config.EVALUATION_DIR, filename)

    @app.errorhandler(413)
    def too_large(_error):
        return error(f'Image is too large. The limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB.', 413, 'too_large')

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith(('/api/', '/predict')):
            return error(exc.description or exc.name, exc.code, exc.name.lower().replace(' ', '_'))
        return exc

    @app.errorhandler(Exception)
    def unhandled(exc):
        log.exception('Unhandled error')
        return error('Internal server error.', 500, 'server_error')

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    create_app().run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
