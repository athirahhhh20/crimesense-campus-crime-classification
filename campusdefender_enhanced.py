from flask import Flask, render_template_string, request, send_from_directory, jsonify, session
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
import base64
import os
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# === Load Models and Encoders ===
try:
    model = joblib.load('final_model.pkl')
    le_crime_type = joblib.load('le_crime_type.pkl')
    le_sex = joblib.load('le_vict_sex.pkl')
    le_area = joblib.load('le_area_name.pkl')
    le_descent = joblib.load('le_vict_descent.pkl')
    print("✅ All models and encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# === Mapping dictionaries - FIXED to match actual encoder classes ===
sex_code_map = {"Female": "F", "Male": "M", "Unknown": "Unknown", "Other": "X"}
area_code_map = {"77th Street": "77th Street", "Mission": "Mission", "Northeast": "Northeast", "Wilshire": "Wilshire"}
descent_code_map = {"Hispanic": "H", "White": "W", "Black": "B", "Unknown": "Unknown", "Asian": "A"}
weapon_mapping = {"No Weapon": 0, "Firearm": 102, "Knife": 400, "Other Weapon": 622}
premis_mapping = {"Residence": 101, "Commercial Building": 108, "Parking Lot": 203, "Street/Sidewalk": 502}
crm_cd_mapping = {"Burglary": 624, "Assault": 740, "Vandalism": 926, "Theft": 330, "Other": 999}
part_mapping = {"Part 1": 1.0, "Part 2": 2.0}

app = Flask(__name__)
app.secret_key = 'campusdefender_secret_key'

# Initialize session for history
def init_session():
    if 'prediction_history' not in session:
        session['prediction_history'] = []
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(APP_ROOT, 'static')

def get_base64_bg(filename):
    try:
        with open(os.path.join(APP_ROOT, filename), 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def create_chart(chart_type, data_col, title):
    try:
        df = pd.read_csv('campus_crime_data.csv')
        
        plt.figure(figsize=(10, 8))
        plt.style.use('default')
        
        if chart_type == 'pie' and data_col in df.columns:
            counts = df[data_col].value_counts().head(8)
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
        elif chart_type == 'bar' and data_col in df.columns:
            counts = df[data_col].value_counts().head(10)
            plt.bar(range(len(counts)), counts.values)
            plt.xticks(range(len(counts)), counts.index, rotation=45)
        elif chart_type == 'hist' and data_col in df.columns:
            df[data_col] = pd.to_numeric(df[data_col], errors='coerce')
            plt.hist(df[data_col].dropna(), bins=30, alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return img_base64
    except:
        return None

def create_location_density_chart():
    try:
        df = pd.read_csv('campus_crime_data.csv')
        
        # Create a more informative chart showing crime distribution
        plt.figure(figsize=(12, 8))
        plt.style.use('default')
        
        # Filter out zero coordinates (invalid data)
        valid_data = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
        
        if len(valid_data) > 0:
            # Create subplots for better analysis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: Geographic crime hotspots (main focus)
            location_counts = valid_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='Crime_Count')
            
            # Create density plot with better visualization
            scatter = ax1.scatter(location_counts['Longitude'], location_counts['Latitude'], 
                                s=location_counts['Crime_Count']*25, alpha=0.7, 
                                c=location_counts['Crime_Count'], cmap='Reds', edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel('Longitude', fontsize=12)
            ax1.set_ylabel('Latitude', fontsize=12)
            ax1.set_title('Geographic Crime Hotspots\n(Size & Color = Crime Frequency)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax1, label='Number of Crimes')
            
            # Right plot: Crime distribution by location zones
            # Create location-based analysis instead of duplicate crime types
            zone_analysis = valid_data.groupby('Crime Type')['Crime Type'].count().sort_values(ascending=True)
            
            bars = ax2.barh(zone_analysis.index, zone_analysis.values, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ax2.set_xlabel('Number of Incidents', fontsize=12)
            ax2.set_title('Crime Frequency Analysis\n(Based on Valid Locations)', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, zone_analysis.values)):
                ax2.text(value + max(zone_analysis.values)*0.01, i, f'{int(value)}', 
                        va='center', fontsize=10, fontweight='bold')
            
        else:
            # Fallback: show crime type distribution only
            crime_counts = df['Crime Type'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            plt.pie(crime_counts.values, labels=crime_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Crime Type Distribution', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Location chart error: {e}")
        return None

def create_crime_statistics_chart():
    """Alternative chart showing crime statistics in bar format"""
    try:
        df = pd.read_csv('campus_crime_data.csv')
        
        plt.figure(figsize=(14, 10))
        
        # Create 2x2 subplot layout for comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Crime Type Bar Chart
        crime_counts = df['Crime Type'].value_counts()
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars1 = ax1.bar(crime_counts.index, crime_counts.values, color=colors1)
        ax1.set_title('Crime Frequency by Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Incidents')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Crime Type Pie Chart
        ax2.pie(crime_counts.values, labels=crime_counts.index, autopct='%1.1f%%', 
               colors=colors1, startangle=90)
        ax2.set_title('Crime Distribution Percentage', fontsize=14, fontweight='bold')
        
        # 3. Location Density (if valid coordinates exist)
        valid_data = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
        if len(valid_data) > 0:
            location_counts = valid_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='Crime_Count')
            scatter = ax3.scatter(location_counts['Longitude'], location_counts['Latitude'], 
                                s=location_counts['Crime_Count']*30, alpha=0.7, 
                                c=location_counts['Crime_Count'], cmap='Reds', edgecolors='black')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('Geographic Crime Hotspots', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Valid\nLocation Data', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=16, color='gray')
            ax3.set_title('Geographic Analysis', fontsize=14, fontweight='bold')
        
        # 4. Total Crime Count Summary
        total_crimes = df['Count'].sum() if 'Count' in df.columns else len(df)
        unique_locations = len(df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].drop_duplicates(['Latitude', 'Longitude']))
        
        stats_text = f"""CRIME STATISTICS SUMMARY
        
Total Incidents: {total_crimes:,}
Unique Crime Types: {len(crime_counts)}
Active Locations: {unique_locations}

Most Common Crime:
{crime_counts.index[0]} ({crime_counts.iloc[0]} incidents)

Crime Distribution:
• {crime_counts.index[0]}: {crime_counts.iloc[0]/total_crimes*100:.1f}%
• {crime_counts.index[1]}: {crime_counts.iloc[1]/total_crimes*100:.1f}%
• Others: {(total_crimes-crime_counts.iloc[0]-crime_counts.iloc[1])/total_crimes*100:.1f}%"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Statistics Overview', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Statistics chart error: {e}")
        return None

def generate_heatmap():
    try:
        df = pd.read_csv('campus_crime_data.csv')
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
        HeatMap(df[['Latitude', 'Longitude']].values).add_to(m)
        return m._repr_html_()
    except:
        return '<p>Heatmap unavailable</p>'

# Enhanced HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Campus Defender: A Crime Classification System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            background: url('data:image/jpg;base64,{{ bg_img }}') center center fixed;
            background-size: cover;
            font-family: 'Montserrat', Arial, sans-serif;
            margin: 0;
            min-height: 100vh;
        }
        .overlay {
            background: rgba(255,255,255,0.65);
            min-height: 100vh;
            padding-bottom: 40px;
        }
        .container {
            background: rgba(255,255,255,0.9);
            border-radius: 18px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
            padding: 30px;
            max-width: 1400px;
            margin: 20px auto;
        }
        h1 {
            background: linear-gradient(90deg, #2c3e50 60%, #2980b9 100%);
            color: #fff;
            padding: 18px 0;
            border-radius: 15px;
            text-align: center;
            font-size: 2.1rem;
            margin-bottom: 18px;
            letter-spacing: 1px;
        }
        h2 {
            color: #2c3e50;
            font-size: 1.5rem;
            margin: 30px 0 15px 0;
            border-bottom: 2px solid #2980b9;
            padding-bottom: 10px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .form-col {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: 600;
            color: #2c3e50;
        }
        label:first-of-type {
            margin-top: 0;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #bfc9d1;
            margin-top: 6px;
            font-size: 1.05rem;
            background: #fff;
            box-sizing: border-box;
        }
        .btn {
            background: linear-gradient(90deg, #2c3e50 60%, #2980b9 100%);
            color: #fff;
            border: none;
            padding: 16px 32px;
            border-radius: 12px;
            margin-top: 20px;
            font-size: 1.15rem;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.2s;
            width: 100%;
        }
        .btn:hover {
            background: linear-gradient(90deg, #2980b9 60%, #2c3e50 100%);
        }
        .result {
            background: #e6f2ff;
            padding: 24px;
            border-radius: 14px;
            border-left: 7px solid #003366;
            margin: 20px 0;
            font-size: 1.15rem;
        }
        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }
        .viz-card {
            background: #fff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .large-viz-card {
            background: #fff;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            grid-column: 1 / -1;
        }
        .viz-card h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.3rem;
            border-bottom: 2px solid #2980b9;
            padding-bottom: 10px;
        }
        .chart-img {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .heatmap {
            grid-column: 1 / -1;
            background: #fff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 5px;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 12px;
            text-align: center;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: #666;
        }
        .tab.active {
            background: #2980b9;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #2980b9;
            font-weight: bold;
            margin-left: 5px;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 280px;
            background-color: #2c3e50;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 12px;
            position: absolute;
            z-index: 1000;
            top: -5px;
            left: 105%;
            font-size: 12px;
            line-height: 1.4;
            font-weight: normal;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 50%;
            right: 100%;
            margin-top: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: transparent #2c3e50 transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .label-with-tooltip {
            display: flex;
            align-items: center;
        }
        .history-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
            border-left: 4px solid #2980b9;
        }
        .history-item {
            background: #fff;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .history-item:first-child {
            border-left: 3px solid #28a745;
        }
        .history-timestamp {
            color: #6c757d;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        .history-inputs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        .history-result {
            background: #e6f3ff;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
            font-weight: bold;
        }
        .clear-history-btn {
            background: #dc3545;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 10px;
        }
        .clear-history-btn:hover {
            background: #c82333;
        }
        @media (max-width: 768px) {
            .viz-grid {
                grid-template-columns: 1fr;
            }
            .container {
                margin: 10px;
                padding: 20px;
            }
            .tooltip .tooltiptext {
                width: 200px;
                left: -100px;
                top: 25px;
            }
            .tooltip .tooltiptext::after {
                top: -5px;
                left: 50%;
                right: auto;
                margin-left: -5px;
                border-color: #2c3e50 transparent transparent transparent;
            }
        }
    </style>
</head>
<body>
    <div class="overlay">
        <div class="container">
            <h1>🛡️ Campus Defender: A Crime Classification System</h1>
            
            <!-- Prediction Form -->
            <h2>🎯 Crime Prediction</h2>
            <form method="POST" id="crimeForm">
                <div class="form-grid">
                    <div class="form-col">
                        <div class="label-with-tooltip">
                            <label for="age">Victim Age:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Age of the crime victim in years. Enter a number between 0 and 120.</span>
                            </span>
                        </div>
                        <input type="number" name="age" id="age" min="0" max="120" placeholder="e.g., 25" required>
                        
                        <div class="label-with-tooltip">
                            <label for="sex_label">Victim Sex:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Biological sex of the victim as recorded in the police report.</span>
                            </span>
                        </div>
                        <select name="sex_label" required>
                            <option value="">Select Sex</option>
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                            <option value="Unknown">Unknown</option>
                            <option value="Other">Other</option>
                        </select>
                        
                        <div class="label-with-tooltip">
                            <label for="time_occ">Time Occurred:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Time when the crime occurred in 24-hour format (HHMM). Examples: 0800 = 8:00 AM, 1430 = 2:30 PM, 2245 = 10:45 PM. Note: We use 24-hour format only (0000-2359).</span>
                            </span>
                        </div>
                        <input type="number" name="time_occ" id="time_occ" min="0" max="2359" placeholder="e.g., 1430 (24-hour format)" required>
                    </div>
                    
                    <div class="form-col">
                        <div class="label-with-tooltip">
                            <label for="area_label">Area Name:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Police division or patrol area where the crime occurred. Different areas may have different crime patterns.</span>
                            </span>
                        </div>
                        <select name="area_label" required>
                            <option value="">Select Area</option>
                            <option value="77th Street">77th Street</option>
                            <option value="Mission">Mission</option>
                            <option value="Northeast">Northeast</option>
                            <option value="Wilshire">Wilshire</option>
                        </select>
                        
                        <div class="label-with-tooltip">
                            <label for="descent_label">Victim Descent:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Ethnic or racial background of the victim as recorded in the police report.</span>
                            </span>
                        </div>
                        <select name="descent_label" required>
                            <option value="">Select Descent</option>
                            <option value="Hispanic">Hispanic</option>
                            <option value="White">White</option>
                            <option value="Black">Black</option>
                            <option value="Unknown">Unknown</option>
                            <option value="Asian">Asian</option>
                        </select>
                        
                        <div class="label-with-tooltip">
                            <label for="weapon_label">Weapon Used:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Type of weapon involved in the crime, if any. This affects crime classification and severity.</span>
                            </span>
                        </div>
                        <select name="weapon_label" required>
                            <option value="">Select Weapon</option>
                            <option value="No Weapon">No Weapon</option>
                            <option value="Firearm">Firearm</option>
                            <option value="Knife">Knife</option>
                            <option value="Other Weapon">Other Weapon</option>
                        </select>
                    </div>
                    
                    <div class="form-col">
                        <div class="label-with-tooltip">
                            <label for="premis_label">Premise:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">Type of location or building where the crime occurred. Different premises have different risk factors.</span>
                            </span>
                        </div>
                        <select name="premis_label" required>
                            <option value="">Select Premise</option>
                            <option value="Residence">Residence</option>
                            <option value="Commercial Building">Commercial Building</option>
                            <option value="Parking Lot">Parking Lot</option>
                            <option value="Street/Sidewalk">Street/Sidewalk</option>
                        </select>
                        
                        <div class="label-with-tooltip">
                            <label for="crm_cd_label">Crime Code:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">General category of crime for initial classification. The system will predict the specific crime type.</span>
                            </span>
                        </div>
                        <select name="crm_cd_label" required>
                            <option value="">Select Crime Code</option>
                            <option value="Burglary">Burglary</option>
                            <option value="Assault">Assault</option>
                            <option value="Vandalism">Vandalism</option>
                            <option value="Theft">Theft</option>
                            <option value="Other">Other</option>
                        </select>
                        
                        <div class="label-with-tooltip">
                            <label for="part_label">Part Classification:</label>
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">FBI Uniform Crime Reporting classification:<br><strong>Part 1:</strong> Serious crimes (murder, rape, robbery, aggravated assault, burglary, theft, arson)<br><strong>Part 2:</strong> Less serious crimes (simple assault, fraud, vandalism, drug offenses)</span>
                            </span>
                        </div>
                        <select name="part_label" required>
                            <option value="">Select Part</option>
                            <option value="Part 1">Part 1 (Serious Crimes)</option>
                            <option value="Part 2">Part 2 (Less Serious Crimes)</option>
                        </select>
                        
                        <button type="submit" class="btn">🔮 Predict Crime Type</button>
                    </div>
                </div>
            </form>
            
            {% if result %}
            <div class="result">
                <h3>🎯 Prediction Result:</h3>
                <p><strong>Predicted Crime Type:</strong> {{ result.pred_label }}</p>
                <p><strong>Confidence:</strong> {{ result.pred_proba }}%</p>
            </div>
            {% endif %}
            
            <!-- History Section -->
            {% if history and history|length > 0 %}
            <div class="history-section">
                <h3>📜 Recent Predictions (Last 10)</h3>
                <button onclick="clearHistory()" class="clear-history-btn">Clear History</button>
                {% for item in history %}
                <div class="history-item">
                    <div class="history-timestamp">
                        {{ item.timestamp }} {% if loop.index == 1 %}(Most Recent){% endif %}
                    </div>
                    <div class="history-inputs">
                        <div><strong>Age:</strong> {{ item.inputs.age }}</div>
                        <div><strong>Sex:</strong> {{ item.inputs.sex_label }}</div>
                        <div><strong>Time:</strong> {{ item.inputs.time_occ }}</div>
                        <div><strong>Area:</strong> {{ item.inputs.area_label }}</div>
                        <div><strong>Descent:</strong> {{ item.inputs.descent_label }}</div>
                        <div><strong>Weapon:</strong> {{ item.inputs.weapon_label }}</div>
                        <div><strong>Premise:</strong> {{ item.inputs.premis_label }}</div>
                        <div><strong>Crime Code:</strong> {{ item.inputs.crm_cd_label }}</div>
                        <div><strong>Part:</strong> {{ item.inputs.part_label }}</div>
                    </div>
                    <div class="history-result">
                        🎯 Predicted: {{ item.result.pred_label }} ({{ item.result.pred_proba }}% confidence)
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <!-- Analytics Dashboard -->
            <h2>📊 Crime Analytics Dashboard</h2>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('overview')">Overview</button>
                <button class="tab" onclick="showTab('temporal')">Enhanced Analysis</button>
                <button class="tab" onclick="showTab('location')">Heatmap</button>
            </div>
            
            <div id="overview" class="tab-content active">
                <div class="viz-grid">
                    <div class="viz-card">
                        <h3>🔍 Crime Type Distribution</h3>
                        {% if crime_pie %}
                        <img src="data:image/png;base64,{{ crime_pie }}" class="chart-img" alt="Crime Types">
                        {% else %}
                        <p>Generating visualization...</p>
                        {% endif %}
                    </div>
                    
                    <div class="viz-card">
                        <h3>📈 Crime Frequency by Type</h3>
                        {% if crime_bar %}
                        <img src="data:image/png;base64,{{ crime_bar }}" class="chart-img" alt="Crime Frequency">
                        {% else %}
                        <p>Generating visualization...</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            

            
            <div id="temporal" class="tab-content">
                <div class="viz-grid">
                    <div class="large-viz-card">
                        <h3>📊 Enhanced Crime Analysis</h3>
                        <p style="color: #6c757d; margin-bottom: 15px;">Comprehensive view of crime patterns and geographic distribution</p>
                        {% if location_analysis %}
                        <img src="data:image/png;base64,{{ location_analysis }}" class="chart-img" alt="Enhanced Crime Analysis">
                        {% else %}
                        <p>Generating visualization...</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <div id="location" class="tab-content">
                <div class="heatmap">
                    <h3>🗺️ Crime Location Heatmap</h3>
                    {{ heatmap_html | safe }}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function clearHistory() {
            if (confirm('Are you sure you want to clear all prediction history?')) {
                fetch('/clear_history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    }
                });
            }
        }
    </script>
</body>
</html>
'''

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['prediction_history'] = []
    return jsonify({'success': True})

@app.route('/', methods=['GET', 'POST'])
def index():
    init_session()  # Initialize session
    result = None
    error_message = None
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Extract and validate form data
            age = int(request.form['age'])
            if not (0 <= age <= 120):
                raise ValueError('Victim Age must be between 0 and 120.')
            
            time_occ = int(request.form['time_occ'])
            if not (0 <= time_occ <= 2359):
                raise ValueError('Time Occurred must be between 0000 and 2359.')
            
            sex_label = request.form['sex_label'].strip()
            area_label = request.form['area_label'].strip()
            descent_label = request.form['descent_label'].strip()
            weapon_label = request.form['weapon_label'].strip()
            premis_label = request.form['premis_label'].strip()
            crm_cd_label = request.form['crm_cd_label'].strip()
            part_label = request.form['part_label'].strip()
            
            # Validate required fields are not empty
            if not all([sex_label, area_label, descent_label, weapon_label, premis_label, crm_cd_label, part_label]):
                raise ValueError('All fields must be selected/filled.')
                
            # Validate keys exist in mappings
            if sex_label not in sex_code_map:
                raise ValueError(f'Invalid sex value: "{sex_label}". Must be one of: {list(sex_code_map.keys())}')
            if area_label not in area_code_map:
                raise ValueError(f'Invalid area value: "{area_label}". Must be one of: {list(area_code_map.keys())}')
            if descent_label not in descent_code_map:
                raise ValueError(f'Invalid descent value: "{descent_label}". Must be one of: {list(descent_code_map.keys())}')
            
            # DEBUG: Print received form values
            print(f"🔍 DEBUG - Received form values:")
            print(f"  sex_label: '{sex_label}'")
            print(f"  area_label: '{area_label}'")
            print(f"  descent_label: '{descent_label}'")
            print(f"  Available sex_code_map keys: {list(sex_code_map.keys())}")
            print(f"  Available area_code_map keys: {list(area_code_map.keys())}")
            print(f"  Available descent_code_map keys: {list(descent_code_map.keys())}")
            
            # Create input DataFrame for prediction - FIXED to use encoders correctly
            # First convert form values to encoder-expected values, then transform
            sex_code = sex_code_map[sex_label]  # 'Female' -> 'F'
            area_code = area_code_map[area_label]  # 'Mission' -> 'Mission'
            descent_code = descent_code_map[descent_label]  # 'Hispanic' -> 'H'
            
            input_df = pd.DataFrame([{
                'Vict Age': age,
                'Vict Sex': le_sex.transform([sex_code])[0],
                'TIME OCC': time_occ,
                'AREA NAME': le_area.transform([area_code])[0],
                'Weapon Used Cd': weapon_mapping[weapon_label],
                'Premis Cd': premis_mapping[premis_label],
                'Vict Descent': le_descent.transform([descent_code])[0],
                'Crm Cd': crm_cd_mapping[crm_cd_label],
                'Part 1-2': part_mapping[part_label]
            }])
            
            # Make prediction
            pred_class = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df).max() * 100
            
            # Get human-readable crime type
            if hasattr(le_crime_type, 'classes_'):
                pred_label = le_crime_type.classes_[pred_class]
            else:
                pred_label = str(pred_class)
            
            result = {
                'pred_label': pred_label,
                'pred_proba': f"{pred_proba:.2f}"
            }
            
            # Add to history (keep only last 10)
            from datetime import datetime
            history_item = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'inputs': {
                    'age': age,
                    'sex_label': sex_label,
                    'time_occ': time_occ,
                    'area_label': area_label,
                    'descent_label': descent_label,
                    'weapon_label': weapon_label,
                    'premis_label': premis_label,
                    'crm_cd_label': crm_cd_label,
                    'part_label': part_label
                },
                'result': result
            }
            
            if 'prediction_history' not in session:
                session['prediction_history'] = []
            
            session['prediction_history'].insert(0, history_item)  # Add to beginning
            session['prediction_history'] = session['prediction_history'][:10]  # Keep only last 10
            session.permanent = True
            
        except Exception as e:
            error_message = str(e)
            result = {'pred_label': 'Error', 'pred_proba': str(e)}
    
    # Generate visualizations
    crime_pie = create_chart('pie', 'Crime Type', 'Crime Type Distribution')
    crime_bar = create_chart('bar', 'Crime Type', 'Crime Frequency by Type')
    # Create a location-based analysis instead of time (since TIME OCC column doesn't exist)
    location_analysis = create_location_density_chart()
    
    # Generate heatmap
    heatmap_html = generate_heatmap()
    
    # Get background image
    bg_img = get_base64_bg('campus_bg.jpg')
    
    # Get history from session
    history = session.get('prediction_history', [])
    
    return render_template_string(HTML_TEMPLATE,
                                result=result,
                                bg_img=bg_img,
                                heatmap_html=heatmap_html,
                                error_message=error_message,
                                crime_pie=crime_pie,
                                crime_bar=crime_bar,
                                location_analysis=location_analysis,
                                history=history)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Campus Defender Flask App...")
    print("📊 Features: Crime Prediction + Analytics Dashboard")
    print("🌐 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 