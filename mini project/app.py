from flask import Flask, render_template, request
import numpy as np
import pickle 

app = Flask(__name__)
# Load the KNN classifier from the pickle file
# with open('etc1.pkl','rb') as file:
#     model=pickle.load(file)

model=pickle.load(open('model.pkl','rb'))

age = None
creatinine = None
gender=None
# Define a function to preprocess user input and make predictions
# Define a function to calculate severity based on age and serum creatinine
def calculate_severity(age, gender, creatinine_mgdl):
    factor = 141 if creatinine_mgdl <= 0.7 else 144
    if gender.lower() == 'male':
        factor = factor if creatinine_mgdl <= 0.7 else 144
    else:
        factor = factor if creatinine_mgdl <= 0.7 else 144
        
    if creatinine_mgdl <= 0.7:
        egfr = (factor * ((creatinine_mgdl / 0.9) ** -0.411)) * ((min(age / 60, 1)) ** 0.411) * ((max(age / 60, 1)) ** -0.329)
    else:
        egfr = (factor * ((creatinine_mgdl / 0.9) ** -1.209)) * ((min(age / 60, 1)) ** 0.411) * ((max(age / 60, 1)) ** -0.329)
    if egfr >= 90:
        stage= "At present you are at Stage 1 (Possible kidney damage (e.g., protein in the urine) with normal kidney function stage 1)"
        precautions="""it's important to take precautions to protect your kidney health and prevent further damage. Here are some precautions you can take:

<br><b>1. Monitor Blood Pressure:</b> High blood pressure can further damage your kidneys. Keep your blood pressure under control by following a low-sodium diet, exercising regularly, and taking any prescribed blood pressure medications as directed by your healthcare provider.

<br><b>2. Manage Blood Sugar Levels:</b> If you have diabetes, it's crucial to keep your blood sugar levels within a healthy range. Follow your diabetes management plan, which may include monitoring your blood sugar levels regularly, taking medications as prescribed, and making dietary and lifestyle changes.

<br><b>3. Stay Hydrated:</b> Drink plenty of water throughout the day to help maintain kidney function and prevent the formation of kidney stones. However, if you have any specific fluid restrictions recommended by your healthcare provider, follow them closely.

<br><b>4. Maintain a Healthy Diet:</b> Eat a balanced diet that is low in processed foods, saturated fats, and added sugars. Include plenty of fruits, vegetables, whole grains, and lean proteins in your meals. Limit your intake of red meat and processed meats.

<br><b>5.Limit Alcohol and Caffeine:</b> Excessive alcohol and caffeine consumption can put extra strain on your kidneys. Limit your intake of alcoholic beverages and caffeinated drinks.

<br><b>6. Quit Smoking:</b> Smoking can worsen kidney function and increase the risk of kidney disease. If you smoke, take steps to quit, and avoid exposure to secondhand smoke.

<br><b>7.Avoid Nephrotoxic Substances:</b> Certain medications, chemicals, and substances can be harmful to the kidneys. Talk to your healthcare provider about any medications you are taking and whether they could be affecting your kidney health. Avoid over-the-counter pain relievers like ibuprofen and naproxen unless specifically advised by your doctor.

<br><b>8. Exercise Regularly:</b> Engage in regular physical activity to maintain a healthy weight and improve overall health. Aim for at least 30 minutes of moderate-intensity exercise most days of the week, as long as it's approved by your healthcare provider.

<br><b>9. Get Regular Check-ups: </b>Visit your healthcare provider regularly for check-ups and monitoring of your kidney function. They can perform tests such as blood tests and urine tests to assess your kidney health and detect any changes early.

<br><b>10. Manage Stress:</b> Chronic stress can contribute to high blood pressure and other risk factors for kidney damage. Practice stress-reduction techniques such as deep breathing, meditation, yoga, or engaging in hobbies you enjoy.

<br>By following these precautions and working closely with your healthcare provider, you can help protect your kidney health and prevent further damage. If you notice any changes in your symptoms or overall health, don't hesitate to contact your healthcare provider for guidance and support.
"""
    elif egfr >= 60:
        stage= "At present you are at stage 2 (Kidney damage with mild loss of kidney function) "
        precautions='''it's crucial to take proactive steps to protect your kidney health and slow down the progression of kidney disease. Here are some precautions you can take:

<br><b>1. Monitor Blood Pressure:</b> High blood pressure can accelerate kidney damage. Keep your blood pressure under control by following a low-sodium diet, exercising regularly, and taking any prescribed blood pressure medications as directed by your healthcare provider.

<br><b>2. Manage Blood Sugar Levels:</b> If you have diabetes, it's essential to keep your blood sugar levels within a healthy range. Follow your diabetes management plan diligently, which may include monitoring your blood sugar levels regularly, taking medications as prescribed, and making dietary and lifestyle changes.

<br><b>3. Maintain a Healthy Diet: </b>Adopt a kidney-friendly diet that is low in sodium, phosphorus, and potassium. Focus on consuming plenty of fruits, vegetables, whole grains, and lean proteins. Limit your intake of processed foods, saturated fats, and added sugars.

<br><b>4. Limit Protein Intake:</b> Excessive protein consumption can strain the kidneys. Consider reducing your intake of high-protein foods and consult with a registered dietitian to determine an appropriate protein intake for your condition.

<br><b>5. Stay Hydrated:</b> Drink an adequate amount of water each day to support kidney function and prevent dehydration. However, if you have any specific fluid restrictions recommended by your healthcare provider, adhere to them closely.

<br><b>6. Quit Smoking:</b> Smoking can worsen kidney function and increase the risk of kidney disease progression. If you smoke, seek support to quit smoking and avoid exposure to secondhand smoke.

<br><b>7. Limit Alcohol and Caffeine:</b> Excessive alcohol and caffeine intake can burden the kidneys. Limit your consumption of alcoholic beverages and caffeinated drinks.

<br><b>8. Exercise Regularly:</b> Engage in regular physical activity to maintain a healthy weight and improve overall health. Aim for at least 30 minutes of moderate-intensity exercise most days of the week, unless advised otherwise by your healthcare provider.

<br><b>9. Manage Medications:</b> Review all medications you're taking with your healthcare provider to ensure they're kidney-safe. Avoid medications known to be nephrotoxic and follow your provider's instructions regarding dosage and frequency.

<br><b>10. Regular Monitoring:</b> Attend regular check-ups with your healthcare provider to monitor your kidney function closely. They may perform tests such as blood tests and urine tests to assess your kidney health and adjust your treatment plan accordingly.

<br><b>11. Manage Stress: </b>Chronic stress can exacerbate health conditions, including kidney disease. Practice stress-reduction techniques such as deep breathing, meditation, or yoga to promote overall well-being.

<br>By implementing these precautions and working closely with your healthcare team, you can help manage kidney damage and slow down the progression of kidney disease. If you have any concerns or experience new symptoms, don't hesitate to contact your healthcare provider for guidance and support.'''
    elif egfr >= 45:
        stage= "At present you are at stage 3a(Mild to moderate loss of kidney function)"
        precautions='''At stage 3a of chronic kidney disease (CKD), it's crucial to take precautions to slow down the progression of kidney damage and manage associated risks. Here are some precautions:

<br><b>1. Monitor Blood Pressure: </b>Keep blood pressure under control. Aim for a target blood pressure of less than 130/80 mmHg. Medications, lifestyle changes, and regular check-ups with a healthcare provider can help achieve this.

<br><b>2. Manage Blood Glucose Levels: </b>If you have diabetes, it's essential to manage your blood sugar levels. Consistently high blood sugar can further damage the kidneys.

<br><b>3. Control Blood Cholesterol Levels:</b> High cholesterol levels can contribute to kidney damage. Follow a heart-healthy diet, exercise regularly, and take prescribed medications to manage cholesterol levels.

<br><b>4. Limit Sodium Intake: </b>Too much sodium can increase blood pressure and worsen kidney function. Aim to limit sodium intake by avoiding processed foods, canned soups, and salty snacks.

<br><b>5. Monitor Protein Intake:</b> While protein is essential for the body, consuming too much can strain the kidneys. Work with a dietitian to determine the right amount of protein for your needs.

<br><b>6. Stay Hydrated:</b> Drink plenty of water unless advised otherwise by your healthcare provider. Proper hydration can help maintain kidney function.

<br><b>7. Quit Smoking: </b>Smoking can worsen kidney function and increase the risk of kidney disease progression. Seek support and resources to quit smoking if you're a smoker.

<br><b>8. Exercise Regularly:</b> Regular physical activity can help control blood pressure, manage weight, and improve overall health. Consult with your healthcare provider before starting any new exercise regimen.

<br><b>9. Limit Alcohol Consumption:</b> Excessive alcohol consumption can affect kidney function and interact with medications. Limit alcohol intake or avoid it altogether.

<br>10. Follow Medication Instructions:</b> Take medications exactly as prescribed by your healthcare provider. Some medications may need to be adjusted based on kidney function.

<br><b>11. Regular Follow-up Visits: </b>Attend regular check-ups with your healthcare provider to monitor kidney function, blood pressure, and overall health. Early detection and management of any issues can help prevent further kidney damage.

<br><b>12. Manage Stress:</b> Chronic stress can affect overall health, including kidney function. Practice stress-reducing techniques such as deep breathing, meditation, or yoga.

<br>Always consult with your healthcare provider before making any significant changes to your diet, exercise routine, or medication regimen. They can provide personalized advice based on your individual health needs and condition.'''
    elif egfr >= 30:
        stage= "At present you are at stage 3a (Moderate to severe loss of kidney function)"
        precautions='''

<br><b>1. Strict Blood Pressure Control: </b>Maintain blood pressure below 130/80 mmHg through medication adherence, lifestyle changes, and regular monitoring.

<br><b>2. Tightly Manage Blood Sugar:</b> Keep blood glucose levels within target range to prevent further kidney damage, especially if diabetic.

<br><b>3. Monitor and Control Cholesterol Levels: </b>Follow a heart-healthy diet, exercise regularly, and take prescribed medications to manage cholesterol levels.

<br><b>4. Limit Sodium Intake:</b> Reduce sodium consumption to less than 2,300 mg per day to manage blood pressure and kidney function.

<br><b>5. Monitor Protein Intake:</b> Consult with a dietitian to determine optimal protein intake to reduce strain on the kidneys.

<br><b>6. Stay Adequately Hydrated:</b> Drink enough water unless advised otherwise by your healthcare provider.

<br><b>7. Quit Smoking Completely: </b>Smoking cessation is crucial to prevent further kidney damage and disease progression.

<br><b>8. Moderate or Eliminate Alcohol:</b> Limit alcohol consumption or avoid it completely to reduce stress on the kidneys.

<br><b>9. Engage in Regular Exercise: </b>Incorporate regular physical activity to manage weight, blood pressure, and overall health.

<br><b>10. Adhere Strictly to Medication Regimen: </b>Take prescribed medications exactly as directed by healthcare providers to manage CKD and associated conditions.

<br><b>11. Practice Stress Management Techniques:</b> Incorporate stress-reducing activities like mindfulness, meditation, or yoga into daily routine.

<br><b>12. Attend Regular Follow-up Appointments:</b> Keep scheduled check-ups with healthcare providers to monitor kidney function and overall health status.

<br>Following these precise precautions can help slow down the progression of CKD and reduce the risk of complications. Always consult with healthcare providers for personalized advice and adjustments to your treatment plan.'''
    elif egfr >= 15:
        stage= "At present you are at stage 4 (Severe loss of kidney function)"
        precautions='''<br><b>1. Strict Blood Pressure Control: </b>Maintain blood pressure below 130/80 mmHg through medication adherence, lifestyle changes, and regular monitoring.

<br><b>2. Tightly Manage Blood Sugar:</b> Keep blood glucose levels within target range to prevent further kidney damage, especially if diabetic.

<br><b>3. Monitor and Control Cholesterol Levels: </b>Follow a heart-healthy diet, exercise regularly, and take prescribed medications to manage cholesterol levels.

<br><b>4. Limit Sodium Intake:</b> Reduce sodium consumption to less than 2,300 mg per day to manage blood pressure and kidney function.

<br><b>5. Monitor Protein Intake:</b> Consult with a dietitian to determine optimal protein intake to reduce strain on the kidneys.

<br><b>6. Stay Adequately Hydrated:</b> Drink enough water unless advised otherwise by your healthcare provider.

<br><b>7. Quit Smoking Completely: </b>Smoking cessation is crucial to prevent further kidney damage and disease progression.

<br><b>8. Moderate or Eliminate Alcohol:</b> Limit alcohol consumption or avoid it completely to reduce stress on the kidneys.

<br><b>9. Engage in Regular Exercise: </b>Incorporate regular physical activity to manage weight, blood pressure, and overall health.

<br><b>10. Adhere Strictly to Medication Regimen: </b>Take prescribed medications exactly as directed by healthcare providers to manage CKD and associated conditions.

<br><b>11. Practice Stress Management Techniques:</b> Incorporate stress-reducing activities like mindfulness, meditation, or yoga into daily routine.

<br><b>12. Attend Regular Follow-up Appointments:</b> Keep scheduled check-ups with healthcare providers to monitor kidney function and overall health status.

<br>Following these precise precautions can help slow down the progression of CKD and reduce the risk of complications. Always consult with healthcare providers for personalized advice and adjustments to your treatment plan.'''
    else:
        stage= "you are at last stage"
        precautions="The one and only precaution is that you have to transplant your kidney"
    return stage,precautions
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    global age, creatinine,gender
    if request.method == 'POST':
        # Get user inputs from the form
        d0=request.form['gender']
        d1 = request.form['a']
        d2 = request.form['b']
        d3 = request.form['c']
        d4 = request.form['d']
        d5 = request.form['e']
        d6 = request.form['f']
        d7 = request.form['g']
        d8 = request.form['h']
        d9 = request.form['i']
        d10 = request.form['j']
        d11 = request.form['k']
        d12 = request.form['l']
        d13 = request.form['m']
        d14 = request.form['n']
        d15 = request.form['o']
        d16 = request.form['p']
        d17 = request.form['q']
        d18 = request.form['r']
        d19 = request.form['s']
        d20 = request.form['t']
        d21 = request.form['u']
        d22 = request.form['v']
        d23 = request.form['w']
        d24= request.form['x']
        arr=np.array([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24]])
        # Extract age and serum creatinine from the form data
        age=float(d1)
        creatinine=float(d12)
        gender=d0
        # Make predictions using the KNN classifier
        predicted_class = model.predict(arr)
        print(predicted_class)
        # Calculate severity based on age and serum creatinine
        severity,precautions = calculate_severity(age, gender,creatinine)
        # Return the predicted class value as a response
        return render_template('result.html', prediction=predicted_class, severity=severity,precaution=precautions)

@app.route('/severity')
def severity():
    global age, creatinine,gender
    severity_value, precautions = calculate_severity(age,gender,creatinine)
    return render_template('severity.html', severity=severity_value,precautions= precautions)
@app.route('/prdt')
def prdt():
    return render_template('prdt.html')
@app.route('/egfr')
def egfr():
    return render_template('egfr.html')
@app.route('/precautions')
def precautions():
    return render_template('precautions.html')
@app.route('/regulatory')
def regulatory():
    return render_template('regulatory.html')
@app.route('/indication')
def indication():
    return render_template('indication.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__':
    app.run(debug=True)
    