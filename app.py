from flask import Flask, render_template, request, jsonify, session, redirect, send_from_directory
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import joblib
import os

app = Flask(__name__)
app.secret_key = 'student_ml_key'

import json

# Global models
model = None  # subject model
lifestyle_model = None

def train_models():
    global model, lifestyle_model
    
    # Train subject model from data1.csv
    try:
        df = pd.read_csv('data1.csv')
        if len(df) > 3:
            X = df[['Internals', 'Externals', 'Credits']].fillna(0)
            y = df['SGPA'].fillna(0)
            model = RandomForestRegressor(n_estimators=10)
            model.fit(X, y)
            print("Subject model trained.")
        else:
            print("Insufficient data1.csv for subject model.")
    except Exception as e:
        print(f"Subject model error: {e}")
    
    # Train lifestyle model
    try:
        df_life = pd.read_csv('lifestyle_data.csv')
        X_life = df_life[['hours', 'sleep', 'attendance']]
        y_life = df_life['score']
        lifestyle_model = RandomForestRegressor(n_estimators=50)
        lifestyle_model.fit(X_life, y_life)
        print("Lifestyle model trained.")
    except Exception as e:
        print(f"Lifestyle model error: {e}")
    
    # Save models
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump({'subject': model, 'lifestyle': lifestyle_model}, f)
    except:
        pass



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/frontend/<path:filename>')
def frontend(filename):
    return send_from_directory('static/frontend', filename)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
        if username in users and users[username] == password:
            session['user'] = username
            return redirect('/')
        else:
            error = 'Invalid credentials'
    except:
        error = 'Login error. Try "Soham Saha" / "Soham123"'
    
    return render_template('index.html', 
                         is_logged_in=False,
                         error=error,
                         prediction=None,
                         hours_msg='', sleep_msg='', attendance_msg='', suggestion='')


@app.route('/demo')
def demo():
    sample_subjects = [{"name": "Math", "internals": 20, "externals": 65, "credits": 4}, {"name": "Physics", "internals": 18, "externals": 62, "credits": 4}]
    
    temp_df = pd.DataFrame(sample_subjects)
    temp_df['Total_Marks'] = (temp_df['internals'] / 25 * 40) + (temp_df['externals'] / 75 * 60)
    temp_df['Grade_Point'] = temp_df['Total_Marks'].apply(lambda x: 10 if x>=90 else 9 if x>=80 else 8 if x>=70 else 7 if x>=60 else 6 if x>=50 else 5 if x>=40 else 0)
    
    total_gp = (temp_df['Grade_Point'] * temp_df['credits']).sum()
    total_c = temp_df['credits'].sum()
    sgpa = total_gp / total_c if total_c > 0 else 0

    # Train simple model if data1.csv has data
    try:
        df = pd.read_csv('data1.csv')
        if len(df) > 3:
            X = df[['Internals', 'Externals', 'Credits']].fillna(0)
            y = df['SGPA'].fillna(df['Grade_Point']).fillna(0)
            model = RandomForestRegressor(n_estimators=10)
            model.fit(X, y)
            pred = model.predict([[22,67,4]])[0]
        else:
            pred = sgpa + 0.3
    except:
        pred = sgpa + 0.3

    # Generate plots
    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    # Bar marks
    axs[0].bar(temp_df['name'], temp_df['Total_Marks'], color='skyblue')
    axs[0].axhline(88, color='red', linestyle='--', label='Potential')
    axs[0].set_title('Marks Analysis')
    axs[0].legend()

    # Pie grades
    grades = temp_df['Grade_Point'].value_counts()
    axs[1].pie(grades.values, labels=[f'GP{f"{g:.0f}" }' for g in grades.index], autopct='%1.1f%%')
    axs[1].set_title('Grade Distribution')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return jsonify({
        'status': 'Demo analysis',
        'sgpa': round(sgpa,2),
        'predicted': round(pred,2),
        'plot_base64': plot_url,
        'data_preview': temp_df.to_dict('records')
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        subjects = data['subjects']
        
        temp_df = pd.DataFrame(subjects)
        temp_df['Total_Marks'] = (temp_df['internals'] / 25 * 40) + (temp_df['externals'] / 75 * 60)
        temp_df['Grade_Point'] = temp_df['Total_Marks'].apply(lambda x: 10 if x>=90 else 9 if x>=80 else 8 if x>=70 else 7 if x>=60 else 6 if x>=50 else 5 if x>=40 else 0)
        
        total_gp = (temp_df['Grade_Point'] * temp_df['credits']).sum()
        total_c = temp_df['credits'].sum()
        sgpa = total_gp / total_c if total_c > 0 else 0
        
        # Prediction
        pred_features = np.array([[22,67,4]])
        if model:
            predicted_gp = model.predict(pred_features)[0]
        else:
            predicted_gp = sgpa + 0.3

        # Plot
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.bar(temp_df['name'], temp_df['Total_Marks'])
        plt.axhline(88, color='r', ls='--', label='Recommended')
        plt.title('Your Marks vs Recommended')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.subplot(1,2,2)
        grades = temp_df['Grade_Point'].value_counts()
        plt.pie(grades.values, labels=[f'GP {g}' for g in grades.index], autopct='%1.1f%%')
        plt.title('Grade Distribution')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({
            'sgpa': round(sgpa,2),
            'predicted': round(predicted_gp,2),
            'plot': plot_url,
            'insights': f'SGPA: {sgpa:.2f} | Avg Marks: {temp_df["Total_Marks"].mean():.1f}% | Predicted: {predicted_gp:.2f} GP'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    train_models()
    print("🌟 Student Analyzer ready on http://127.0.0.1:5000 (Neon UI + ML)")
    print("💡 Login: 'Soham Saha' / 'Soham123' or signup")
    print("📊 /demo for API demo, /analyze POST for subjects")
    app.run(debug=True, port=5000)
