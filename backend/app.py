"""Validated single and batch inference using the serialized training pipeline."""
from pathlib import Path
import hashlib
import json
import logging
import time
import uuid
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, g
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / 'superkart_model.joblib')
metadata = json.loads((ROOT / 'model_metadata.json').read_text())
FEATURES = metadata['features']
NUMERIC = metadata['numeric_features']
CATEGORIES = {
    'Product_Sugar_Content': ['Low Sugar', 'Regular', 'No Sugar'],
    'Store_Size': ['Small', 'Medium', 'High'],
    'Store_Location_City_Type': ['Tier 1', 'Tier 2', 'Tier 3'],
    'Store_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Departmental Store', 'Food Mart'],
    'Product_Id_char': ['FD', 'DR', 'NC'],
    'Product_Type_Category': ['Perishables', 'Non Perishables'],
}
superkart_api = Flask('SuperKart')
superkart_api.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

def validate(frame):
    """Reject malformed input; warn separately about valid but unfamiliar data."""
    missing, extra = set(FEATURES) - set(frame), set(frame) - set(FEATURES)
    if missing or extra:
        raise ValueError(f'Missing columns: {sorted(missing)}; unexpected columns: {sorted(extra)}')
    if not 1 <= len(frame) <= 10000:
        raise ValueError('Provide between 1 and 10000 rows.')
    frame = frame[FEATURES].copy()
    if frame.isna().any().any():
        raise ValueError('Missing values are not allowed.')
    for col in NUMERIC:
        if frame[col].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f'{col} must be numeric, not boolean.')
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} must contain finite numbers.')
    if (frame[['Product_Weight', 'Product_MRP']] <= 0).any().any():
        raise ValueError('Product weight and MRP must be positive.')
    if not frame['Product_Allocated_Area'].between(0, 1).all():
        raise ValueError('Allocated area must be between 0 and 1.')
    age = frame['Store_Age_Years']
    if (age < 0).any() or (age % 1 != 0).any():
        raise ValueError('Store age must be a nonnegative integer.')
    frame['Product_Sugar_Content'] = frame['Product_Sugar_Content'].replace({'reg': 'Regular'})
    for col, allowed in CATEGORIES.items():
        if not frame[col].isin(allowed).all():
            raise ValueError(f'Invalid category in {col}. Allowed: {allowed}')
    notices = []
    for i, row in frame.iterrows():
        reasons = []
        for col in NUMERIC:
            bounds = metadata['training_ranges'][col]
            if not bounds['min'] <= float(row[col]) <= bounds['max']:
                reasons.append(f'{col} outside training range')
        for col in CATEGORIES:
            if row[col] not in metadata['training_categories'][col]:
                reasons.append(f'{col} unseen in training')
        if reasons:
            notices.append({'row': int(i), 'warnings': reasons})
    return frame, notices

@superkart_api.before_request
def begin_request():
    g.started = time.perf_counter()
    g.request_id = uuid.uuid4().hex

@superkart_api.after_request
def trace(response):
    response.headers['X-Request-ID'] = g.request_id
    response.headers['X-Model-ID'] = metadata['model_sha256'][:16]
    logging.info(json.dumps({'request_id': g.request_id, 'method': request.method,
        'path': request.path, 'status': response.status_code,
        'latency_ms': round((time.perf_counter()-g.started)*1000, 2),
        'model_id': metadata['model_sha256'][:16]}))
    return response

@superkart_api.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description), error.code

@superkart_api.get('/')
def home():
    return jsonify(service='SuperKart', status='ready', model_id=metadata['model_sha256'][:16])

@superkart_api.post('/v1/predict')
def predict_sales():
    if not request.is_json:
        return jsonify(error='Send a JSON object.'), 415
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify(error='Expected one JSON object.'), 400
    try:
        frame, notices = validate(pd.DataFrame([data]))
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400
    prediction = float(model.predict(frame)[0])
    return jsonify(Sales=prediction, warnings=notices)

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    if 'file' not in request.files:
        return jsonify(error='Upload a CSV using the multipart field file.'), 400
    try:
        frame = pd.read_csv(request.files['file'])
        frame, notices = validate(frame)
    except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify(error=str(error)), 400
    predictions = model.predict(frame)
    response = jsonify({str(i): round(float(p), 2) for i, p in enumerate(predictions)})
    # Keep the course's row-index response contract. Warnings are summarized in a header.
    response.headers['X-Input-Warning-Rows'] = ','.join(str(n['row']) for n in notices)
    return response

if __name__ == '__main__':
    superkart_api.run(host='127.0.0.1', port=7860, debug=False)
