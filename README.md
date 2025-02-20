# Improving-data-accuracy-in-CRM-using-AI
This is a Flask-based CRM Data Analysis Web Application that allows users to upload CSV files, perform data cleaning, detect outliers, visualize data using charts, and export processed data in various formats.

Features

Upload CSV files: and perform basic data validation.
Data Cleaning: Handles missing values and removes duplicates.
Outlier Detection: Uses Z-score to detect outliers in numerical data.
Data Visualization: Supports bar, line, scatter, histogram, box, and violin plots.
Correlation Analysis: Generates a heatmap to show relationships between numerical features.
Export Processed Data: Supports CSV, Excel, and JSON formats.
Technologies Used

Backend: Flask, Pandas, NumPy, HTML, CSS, JavaScript
Visualization: Matplotlib, Seaborn
Data Processing: Scipy, Pandas
Installation Prerequisites Ensure you have Python 3.x installed on your system.

Create a virtual environment python -m venv venv source venv/bin/activate # On Windows use: venv\Scripts\activate

Install dependencies pip install -r requirements.txt


Usage
Running the Application
```bash
python app.py
The app will run locally on http://127.0.0.1:5000/.

Navigating the Application

Upload a CSV file on the homepage.
View data insights, including missing values and duplicate entries.
Analyze statistical summaries and detect outliers.
Generate charts and correlation heatmaps.
Export cleaned data in CSV, Excel, or JSON format.
Future Enhancements

Machine Learning Integration** for customer segmentation and churn prediction.
User Authentication for personalized data access.
Database Storage instead of temporary in-memory storage.

Contributors:
Deepasha Shukla - Model Evaluation and Performance Analysis https://github.com/deepasha2305
Deeksha M - model training and visulaiztion https://github.com/deeksha-manjunath2
Akhila Anant Gouda- Data Cleaning and Preprocessing https://github.com/akhilagouda
Saurabh Kumar Gupta- Hyperparameter Optimizations and documentation. https://github.com/guptasaurabh9162
