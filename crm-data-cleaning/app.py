from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats

app = Flask(__name__)

# Global variable to store the DataFrame and upload status
df = None
upload_status = {'message': '', 'type': ''}

@app.route('/')
def index():
    return render_template('upload.html', status=upload_status)

@app.route('/home')
def home():
    if df is None:
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df, upload_status
    if 'file' not in request.files:
        upload_status = {'message': 'No file uploaded', 'type': 'error'}
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        upload_status = {'message': 'No file selected', 'type': 'error'}
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            upload_status = {'message': 'File uploaded successfully!', 'type': 'success'}
            return redirect(url_for('home'))
        except Exception as e:
            upload_status = {'message': f'Error reading file: {str(e)}', 'type': 'error'}
            return redirect(url_for('index'))
    
    upload_status = {'message': 'Invalid file format. Please upload a CSV file.', 'type': 'error'}
    return redirect(url_for('index'))

@app.route('/data')
def data():
    if df is None:
        return redirect(url_for('index'))
    
    # Basic analysis
    analysis = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': len(df[df.duplicated()]),
        'total_rows': len(df),
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Statistical analysis for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats_analysis = {}
    
    for col in numerical_cols:
        stats_analysis[col] = {
            'mean': round(df[col].mean(), 2),
            'median': round(df[col].median(), 2),
            'std': round(df[col].std(), 2),
            'min': round(df[col].min(), 2),
            'max': round(df[col].max(), 2)
        }
    
    # Detect outliers using Z-score
    outliers = {}
    for col in numerical_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = len(z_scores[z_scores > 3])  # Count outliers (|z| > 3)
    
    return render_template('data.html', 
                         analysis=analysis, 
                         stats_analysis=stats_analysis,
                         outliers=outliers)

@app.route('/corrected_data')
def corrected_data():
    if df is None:
        return redirect(url_for('index'))
    
    # Clean the data
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna()  # Remove missing values
    cleaned_df = cleaned_df.drop_duplicates()  # Remove duplicates
    
    return render_template('corrected_data.html', 
                         tables=[cleaned_df.to_html(classes='data table table-striped')], 
                         titles=cleaned_df.columns.values)

@app.route('/charts')
def charts():
    if df is None:
        return redirect(url_for('index'))
    
    return render_template('charts.html', 
                         columns=df.columns.tolist(),
                         numerical_columns=df.select_dtypes(include=[np.number]).columns.tolist())

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        chart_type = request.form.get('chart_type')
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar':
            plt.bar(df[x_column], df[y_column])
        elif chart_type == 'line':
            plt.plot(df[x_column], df[y_column])
        elif chart_type == 'scatter':
            plt.scatter(df[x_column], df[y_column])
        elif chart_type == 'histogram':
            plt.hist(df[x_column], bins=30)
            y_column = 'Frequency'
        elif chart_type == 'box':
            plt.boxplot(df[x_column])
            y_column = 'Value'
        elif chart_type == 'violin':
            sns.violinplot(data=df, y=x_column)
            y_column = 'Value'
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{chart_type.capitalize()} Chart: {x_column} vs {y_column}')
        plt.xticks(rotation=45)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'image': image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/correlation')
def correlation():
    if df is None:
        return redirect(url_for('index'))
    
    numerical_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    correlation_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template('correlation.html', 
                         correlation_plot=correlation_plot,
                         correlation_data=correlation_matrix.to_html(classes='table table-striped'))

@app.route('/export/<format>')
def export_data(format):
    if df is None:
        return redirect(url_for('index'))
    
    buf = io.BytesIO()
    
    if format == 'csv':
        df.to_csv(buf, index=False)
        mimetype = 'text/csv'
        extension = 'csv'
    elif format == 'excel':
        df.to_excel(buf, index=False)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        extension = 'xlsx'
    elif format == 'json':
        df.to_json(buf)
        mimetype = 'application/json'
        extension = 'json'
    else:
        return "Invalid format", 400
    
    buf.seek(0)
    return send_file(
        buf,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f'data_export.{extension}'
    )

if __name__ == '__main__':
    app.run(debug=True)