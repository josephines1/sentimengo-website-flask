import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, render_template, request, send_file, redirect, jsonify, url_for, session, flash
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from nltk.corpus import stopwords
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from tempfile import NamedTemporaryFile
import base64
from datetime import timedelta, datetime

# Flask app initialization
app = Flask(__name__)

# Configurations
app.config['CSV_OUTPUT_FOLDER'] = "C:\\Users\\USER\\Komunitas Maribelajar Indonesia\\CP7 - 07 - Gema Indonesia - Documents\\General\\06 - Deployment\\csv_outputs"
app.config['PDF_REPORT_FOLDER'] = "C:\\Users\\USER\\Komunitas Maribelajar Indonesia\\CP7 - 07 - Gema Indonesia - Documents\\General\\06 - Deployment\\pdf_reports"
app.config['INPUT_FOLDER'] = "C:\\Users\\USER\\Komunitas Maribelajar Indonesia\\CP7 - 07 - Gema Indonesia - Documents\\General\\06 - Deployment\\csv_excel_inputs"
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
app.config['SECRET_KEY'] = '754ea0d6b3be1ef6d0f54226'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'data' not in session:
        session['data'] = None
    if 'preprocessed_data' not in session:
        session['preprocessed_data'] = None
    if 'sentiment_counts' not in session:
        session['sentiment_counts'] = None
    if 'avg_probability' not in session:
        session['avg_probability'] = None
    if 'image_data' not in session:
        session['image_data'] = None

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("josephine-huggingface/bert-sentimengo")
model = AutoModelForSequenceClassification.from_pretrained("josephine-huggingface/bert-sentimengo")
key_norm = pd.read_csv('key_norm.csv')

# Preprocessing functions
def casefolding(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] 
                     if (key_norm['singkat'] == word).any() else word for word in text.split()])
    return str.lower(text)

stopwords_ind = stopwords.words('indonesian')

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    return stemmer.stem(text)

def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

def create_sentiment_graph(df_result):
    # Sentiment distribution count
    sentiment_counts = df_result['sentiment'].value_counts()
    
    # Create a pie chart for sentiment distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
    ax.set_ylabel('')  # Hide the label of the pie chart
    ax.set_title("Sentiment Distribution")
    
    return fig, sentiment_counts

def calculate_average_probability(df_result):
    # Calculate average confidence score
    avg_probability = df_result['Confidence Score'].mean()
    return avg_probability

def save_to_pdf(df, fig, avg_probability, pdf_buffer=None, output_path=None):
    # Buat dokumen PDF
    if pdf_buffer:
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    else:
        doc = SimpleDocTemplate(output_path, pagesize=letter)

    elements = []

    # Tambahkan judul dan rata-rata
    elements.append(Table([[f"Sentiment Analysis Report"]], style=[
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    elements.append(Table([[f"Rata-rata confidence score: {avg_probability:.2f}"]], style=[
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
    ]))

    # Membagi tabel menjadi beberapa bagian
    table_chunks = split_table(df.head(10))  # Ambil 10 baris pertama untuk setiap chunk
    for chunk in table_chunks:
        # Gunakan Paragraph untuk membungkus teks
        data = [[Paragraph(cell, getSampleStyleSheet()['Normal']) if isinstance(cell, str) else cell for cell in row] for row in chunk.values.tolist()]
        table_data = [chunk.columns.tolist()] + data
        
        # Tentukan lebar kolom untuk membatasi lebar
        col_widths = [100] * len(chunk.columns)  # Tentukan lebar kolom (ubah sesuai kebutuhan)
        
        # Buat tabel dengan lebar kolom yang ditentukan
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)

    # Tambahkan grafik ke PDF
    with NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
        fig.savefig(temp_img_file.name, format="png")
        temp_img_path = temp_img_file.name

    elements.append(Table([["Distribusi Sentimen"]], style=[
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
    ]))
    elements.append(Image(temp_img_path, width=400, height=400))

    # Build dokumen PDF
    doc.build(elements)

    # Bersihkan file sementara
    os.remove(temp_img_path)


def split_table(df, max_columns=5):
    """
    Membagi tabel menjadi beberapa bagian jika jumlah kolom melebihi batas.
    """
    num_columns = df.shape[1]
    chunks = [df.iloc[:, i:i + max_columns] for i in range(0, num_columns, max_columns)]
    return chunks

# Sentiment analysis function
def analyze_sentiment(df, content_column):
    df.dropna(subset=[content_column], inplace=True)
    text_preprocessed = df[content_column].apply(text_preprocessing_process)

    inputs = tokenizer(text_preprocessed.tolist(), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1).numpy()
    predicted_labels = np.argmax(probabilities, axis=1)
    sentiment_labels = ["netral" if label == 0 else "positif" if label == 1 else "negatif" for label in predicted_labels]
    df['sentiment'] = sentiment_labels
    df['Confidence Score'] = probabilities.max(axis=1)
    return df

# Allowed file extension check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

# Handling file upload
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('File tidak diperbolehkan!', 'error')
            return redirect(request.url)

        # Simpan file input dari user
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") # Format: YYYYMMDD_HHMMSS

        filename = secure_filename(file.filename)
        new_filename = f"input_{timestamp}_{filename}"

        file_path = os.path.join(app.config['INPUT_FOLDER'], new_filename)
        file.save(file_path)

        # Read file and save to session
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)

        # Define required columns
        required_columns = ['content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'kota']
        session['columns'] = required_columns

        # Check if all required columns are present in the uploaded file
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            flash(f'Kolom berikut tidak ditemukan: {", ".join(missing_columns)}', 'error')
            return redirect(request.url)
        
        # Filter only required columns (columns present in both df and required_columns)
        df = df[required_columns]

        # Store the filtered data in session
        session['data'] = df.to_dict(orient='records')

        # Preprocess data
        df = df.drop_duplicates()
        content_column = request.form.get('content_column', default=df.columns[0])
        if content_column not in df.columns:
            flash(f"Kolom {content_column} tidak ditemukan!", 'error')
            return redirect(request.url)

        df.dropna(subset=[content_column], inplace=True)
        df = df.dropna(subset=[content_column])
        for col, dtype in df.dtypes.items():
            if dtype == 'object':
                df[col].fillna("Unknown", inplace=True)
            elif np.issubdtype(dtype, np.datetime64):
                df[col].fillna(pd.Timestamp("1970-01-01"), inplace=True)

        session['preprocessed_data'] = df.to_dict(orient='records')
        return redirect(url_for('view_data'))

    return render_template('index.html')

# Viewing data
@app.route('/view-data', methods=['GET'])
def view_data():
    if 'preprocessed_data' not in session:
        return redirect(url_for('index'))

    df = pd.DataFrame(session['preprocessed_data'])
    columns = df.columns.tolist()
    values = df.values.tolist()

    return render_template('view-data.html', df=values, columns=columns)

# Analyze and show sentiment result
@app.route('/choose_column', methods=['POST'])
def choose_column():
    content_column = request.form['content_column']
    df = pd.DataFrame(session['preprocessed_data'])
    if content_column not in df.columns:
        return jsonify({'error': 'Column not found'})
    
    session['content_column'] = content_column

    return redirect(url_for('summary'))

# Summary and download options
@app.route('/summary', methods=['GET'])
def summary():
    if 'sentiment_counts' not in session or 'avg_probability' not in session or 'image_data' not in session or 'content_column' not in session:
        return redirect(url_for('index'))
    
    content_column = session['content_column']
    df = pd.DataFrame(session['preprocessed_data'])
    
    df_result = analyze_sentiment(df, content_column)
    sentiment_counts = df_result['sentiment'].value_counts()
    sentiment_counts_dict = sentiment_counts.to_dict()

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['yellow', 'green', 'red'])
    ax.axis('equal')

    img_stream = BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)   

    avg_probability = float(df_result['Confidence Score'].mean())
    avg_probability = round(avg_probability * 100, 2)
    image_data = base64.b64encode(img_stream.getvalue()).decode('utf-8')

    session['sentiment_counts'] = sentiment_counts_dict
    session['avg_probability'] = avg_probability
    session['image_data'] = image_data

    columns = df_result.columns.tolist()
    values = df_result.values.tolist()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Tentukan path untuk PDF dengan nama file unik
    pdf_filename = f"sentiment_analysis_report_{timestamp}.pdf"
    pdf_path = os.path.join(app.config['PDF_REPORT_FOLDER'], pdf_filename)

    # Tentukan path untuk CSV dengan nama file unik
    csv_filename = f"sentiment_analysis_output_{timestamp}.csv"
    csv_path = os.path.join(app.config['CSV_OUTPUT_FOLDER'], csv_filename)

    # Create sentiment graph and save files
    save_to_pdf(df_result, fig, avg_probability, pdf_path)
    df_result.to_csv(csv_path, index=False)

    return render_template('summary.html', 
                           sentiment_counts=sentiment_counts_dict, 
                           avg_probability=avg_probability, 
                           image_data=image_data,
                           df=values, columns=columns)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    if 'preprocessed_data' not in session:
        return redirect(url_for('index'))

    df = pd.DataFrame(session['preprocessed_data'])
    content_column = session['content_column']

    if content_column not in df.columns:
        return f"Kolom {content_column} tidak ditemukan!"

    df_result = analyze_sentiment(df, content_column)

    # Generate sentiment graph and PDF
    fig, sentiment_counts = create_sentiment_graph(df_result)
    avg_probability = calculate_average_probability(df_result)

    # Create PDF in memory (without saving it to disk)
    pdf_buffer = BytesIO()
    save_to_pdf(df_result, fig, avg_probability, pdf_buffer=pdf_buffer)  # PDF to buffer for user download

    # Set buffer position to the beginning
    pdf_buffer.seek(0)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Tentukan path untuk PDF dengan nama file unik
    pdf_filename = f"sentiment_analysis_report_{timestamp}.pdf"

    return send_file(pdf_buffer, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')


@app.route('/download_csv')
def download_csv():
    if 'preprocessed_data' not in session or 'content_column' not in session:
        return redirect(url_for('index'))
    
    content_column = session['content_column']
    df = pd.DataFrame(session['preprocessed_data'])
    df_result = analyze_sentiment(df, content_column)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Tentukan path untuk CSV dengan nama file unik
    csv_filename = f"sentiment_analysis_output_{timestamp}.csv"
    csv_path = os.path.join(app.config['CSV_OUTPUT_FOLDER'], csv_filename)

    df_result.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    # Cek apakah folder untuk CSV sudah ada, jika belum buat folder tersebut
    if not os.path.exists(app.config['CSV_OUTPUT_FOLDER']):
        os.makedirs(app.config['CSV_OUTPUT_FOLDER'])
    
    # Cek apakah folder untuk PDF sudah ada, jika belum buat folder tersebut
    if not os.path.exists(app.config['PDF_REPORT_FOLDER']):
        os.makedirs(app.config['PDF_REPORT_FOLDER'])
    
    # Jalankan aplikasi Flask
    app.run(debug=True)
