from flask import Flask, jsonify, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    try:
        # Validate JSON
        if not request.is_json:
            return jsonify({'error': 'Expected application/json request'}), 400

        payload = request.get_json()
        if 'data' not in payload:
            return jsonify({'error': 'JSON must contain "data" key with feature array'}), 400

        raw_features = payload['data']
        input_array = np.array(raw_features).reshape(1, -1)

        # Check expected feature count if available
        expected = None
        if hasattr(scaler, 'n_features_in_'):
            expected = int(scaler.n_features_in_)
        elif hasattr(model, 'n_features_in_'):
            expected = int(model.n_features_in_)

        if expected is not None and input_array.shape[1] != expected:
            return jsonify({
                'error': 'Wrong number of features',
                'received': input_array.shape[1],
                'expected': expected
            }), 400

        scaled_features = scaler.transform(input_array)
        prediction = model.predict(scaled_features)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Log detailed traceback to server log and return a concise error to client
        import traceback
        tb = traceback.format_exc()
        app.logger.exception('Prediction error')
        return jsonify({'error': str(e), 'trace': tb}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080,debug=True)