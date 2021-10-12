from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html') 


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    #float_features = [float(x) for x in request.form.values()]
    education= request.form['Education']
    age = request.form['Age']
    experience = request.form['Experience']
    us = request.form['United States']
    uk = request.form['United Kingdom']
    swiss = request.form['Switzerland']
    spain = request.form['Spain']
    poland = request.form['Poland']
    neth = request.form['Netherlands']
    italy = request.form['Italy']
    india = request.form['India']
    germany = request.form['Germany']
    france = request.form['France']
    canada = request.form['Canada']
    brazil = request.form['Brazil']

    final_features = np.array([[education, age, experience, us, uk, swiss, spain, poland, neth, italy, india, germany, france, canada, brazil]])
    #final_features = [np.array(float_features)]
    prediction = model.predict(final_features)# 고정시키고 싶은 변수들???)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='당신의 예상 연봉은 $ {} 입니다'.format(output))




if __name__ == "__main__":
    app.run (debug=True)