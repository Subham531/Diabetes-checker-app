from flask import Flask,jsonify,request
import numpy as np
import pickle 

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pickle','rb'))
@app.route('/predict', methods=['POST'])

def predict():
    raw_features = request.json['data']
    input_array = np.array(raw_features).reshape(1,-1)
    scaled_features = scaler.transform(input_array)
    prediction = model.predict(scaled_features)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == __main__:
    app.run(host = '0.0.0.0',port=8080,debug=True)