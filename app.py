import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
finalized_model = pickle.load(open('finalized_model.pkl', 'rb'),encoding='latin1')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    transcription = [x for x in request.form.values()]
    final_transcription = np.array(transcription)
    prediction = finalized_model.predict(final_transcription)

    output = prediction[0]

    return render_template('index.html', prediction_text='Medical Speciality :: {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = finalized_model.predict(np.array(list(data.values())))

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
