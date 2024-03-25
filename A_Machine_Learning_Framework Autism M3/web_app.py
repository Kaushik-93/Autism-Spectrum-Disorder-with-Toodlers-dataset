from flask import Flask,render_template,url_for,request
import pickle
import joblib
import numpy as np

app=Flask(__name__)

model_path = './model.jlb'
model = joblib.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    A1 = int(request.form['A1'])
    A2 = int(request.form['A2'])
    A3 = int(request.form['A3'])
    A4 = int(request.form['A4'])
    A5 = int(request.form['A5'])
    A6 = int(request.form['A6'])
    A7 = int(request.form['A7'])
    A8 = int(request.form['A8'])
    A9 = int(request.form['A9'])
    A10 = int(request.form['A10'])
    Qchat_10_Score = int(request.form['Qchat_10_Score'])
    Sex = int(request.form['Sex'])
    Ethnicity = int(request.form['Ethnicity'])
    Jaundice = int(request.form['Jaundice'])
   

    qurry= np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Qchat_10_Score,Sex,Ethnicity,Jaundice]])
    
    prediction = model.predict(qurry)
    print("************************",prediction)

    
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
