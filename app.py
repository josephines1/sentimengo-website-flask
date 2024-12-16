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
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, HRFlowable, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
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
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=['#16A34A', '#2563EB', '#DC2626'])
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

    # Definisikan style untuk Times New Roman
    style_normal = ParagraphStyle(
        name='Normal', 
        fontName='Times-Roman', 
        fontSize=12, 
        leading=14,
        alignment=4,
        leftIndent=20
    )

    style_bold_and_center = ParagraphStyle(
        name='Bold', 
        fontName='Times-Bold', 
        fontSize=14, 
        leading=16,
        alignment=1
    )

    style_header = ParagraphStyle(
        name='Header', 
        fontName='Times-Bold', 
        fontSize=16, 
        leading=18,
        alignment=1
    )

    logo_path = "./static/images/logo.png"

    elements = []

    # Menambahkan Header (logo + nama proyek)
    # Menambahkan logo di header
    elements.append(Image(logo_path, width=50, height=50))  # Sesuaikan dengan ukuran logo Anda

    # Menambahkan nama proyek "SentimenGo"
    elements.append(Paragraph("<b>SentimenGo</b>", style_header))

    # Tambahkan judul dan rata-rata
    elements.append(Paragraph(f"Sentiment Analysis Report", style_header))
    elements.append(Paragraph(f"Confidence score: {avg_probability:.2f}", style_bold_and_center))

    # Menghitung jumlah absolut untuk setiap sentimen
    positive_count = df['sentiment'].value_counts().get('positif', 0)
    negative_count = df['sentiment'].value_counts().get('negatif', 0)
    neutral_count = df['sentiment'].value_counts().get('netral', 0)

    # Menghitung persentase
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    sentiment_counts = sentiment_counts.round(2)  # Membulatkan persentase ke 2 angka desimal

    positive_percentage = sentiment_counts.get('positif', 0)
    negative_percentage = sentiment_counts.get('negatif', 0)
    neutral_percentage = sentiment_counts.get('netral', 0)

    # Membuat rekomendasi berdasarkan sentimen dominan
    if positive_percentage > negative_percentage and positive_percentage > neutral_percentage:
        recommendation = """
        Rekomendasi:
        <ul>
            <li>Perkuat aspek yang disukai pelanggan dengan mempertahankan atau meningkatkan kualitas layanan yang positif.</li>
            <li>Fokus pada pengembangan produk/layanan yang mendapatkan ulasan positif.</li>
        </ul>
        """
    elif negative_percentage > positive_percentage and negative_percentage > neutral_percentage:
        recommendation = """
        Rekomendasi:
        <ul>
            <li>Lakukan perbaikan pada aspek yang sering dikeluhkan oleh pelanggan, yang menyebabkan ulasan negatif.</li>
            <li>Tingkatkan kualitas layanan di area yang berpotensi mengurangi ulasan negatif.</li>
        </ul>
        """
    elif neutral_percentage > positive_percentage and neutral_percentage > negative_percentage:
        recommendation = """
        Rekomendasi:
        <ul>
            <li>Fokus pada peningkatan pengalaman pengguna agar mereka merasa lebih yakin untuk memberikan penilaian lebih jelas.</li>
            <li>Arahkan ulasan ke arah yang lebih positif atau negatif dengan mengidentifikasi titik perbaikan.</li>
        </ul>
        """
    else:
        # Jika tidak ada yang dominan, beri rekomendasi umum
        recommendation = """
        Rekomendasi:
        <ul>
            <li>Tinjau ulasan secara menyeluruh untuk memastikan kualitas layanan dan produk yang konsisten.</li>
            <li>Tingkatkan keterlibatan dengan pengguna untuk mendapatkan umpan balik yang lebih beragam.</li>
        </ul>
        """

    # Analisis grafik dan rekomendasi
    analysis_text = f"""
    Berdasarkan hasil diagram distribusi sentimen pada ulasan yang diberikan, dapat diketahui bahwa jumlah ulasan positif 
    adalah {positive_count} ({positive_percentage}%), jumlah ulasan negatif adalah {negative_count} ({negative_percentage}%), dan jumlah ulasan netral adalah {neutral_count} ({neutral_percentage}%).
    """

    # Tambahkan grafik ke PDF
    with NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
        fig.savefig(temp_img_file.name, format="png")
        temp_img_path = temp_img_file.name

    # Menambahkan garis pemisah sebelum grafik menggunakan HRFlowable
    elements.append(HRFlowable(width="100%", thickness=1, lineCap='round', color="#000", spaceBefore=1, spaceAfter=1, hAlign='CENTER', vAlign='BOTTOM', dash=None))

    # Menambahkan spacer (ruang kosong) sebelum "Distribusi Sentimen"
    elements.append(Spacer(1, 20))

    # Menambahkan judul untuk grafik
    elements.append(Paragraph("<b>Distribusi Sentimen</b>", style_normal))

    # Menampilkan gambar grafik
    elements.append(Image(temp_img_path, width=200, height=200))

    # Menambahkan analisis teks ke dalam PDF
    elements.append(Paragraph(analysis_text, style_normal))

    # Menambahkan rekomendasi berdasarkan sentimen dominan
    elements.append(Paragraph(recommendation, style_normal))

    # Build dokumen PDF
    doc.build(elements)

    # Bersihkan file sementara
    os.remove(temp_img_path)

# Sentiment analysis function
def analyze_sentiment(df):
    df.dropna(subset=['content'], inplace=True)
    text_preprocessed = df['content'].apply(text_preprocessing_process)

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
    session.clear()
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
        if 'content' not in df.columns:
            flash(f"Kolom 'content' tidak ditemukan!", 'error')
            return redirect(request.url)

        df.dropna(subset=['content'], inplace=True)
        df = df.dropna(subset=['content'])
        for col, dtype in df.dtypes.items():
            if dtype == 'object':
                df[col].fillna("Unknown", inplace=True)
            elif np.issubdtype(dtype, np.datetime64):
                df[col].fillna(pd.Timestamp("1970-01-01"), inplace=True)

        # Validate 'at' column format (YYYY-MM-DD HH:MM:SS)
        try:
            df['at'] = pd.to_datetime(df['at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # Check if any date is invalid (NaT means Not a Time)
            if df['at'].isna().any():
                invalid_rows = df[df['at'].isna()]
                flash(f"Tanggal pada kolom at yang tidak valid: {invalid_rows.index.tolist()}. \nUpload kolom tanggal at dengam format YYYY-MM-DD HH:MM:SS", 'error')
                return redirect(request.url)
        except Exception as e:
            flash(f"Terjadi kesalahan saat memvalidasi kolom 'at': {str(e)}", 'error')
            return redirect(request.url)

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
@app.route('/process_data', methods=['POST'])
def process_data():
    df = pd.DataFrame(session['preprocessed_data'])
    if 'content' not in df.columns:
        return jsonify({'error': 'Column not found'})

    # analisis sentimen
    df_result = analyze_sentiment(df)

    sentiment_counts = df_result['sentiment'].value_counts()

    # membuat grafik
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#16A34A', '#2563EB', '#DC2626'])
    ax.axis('equal')

    img_stream = BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)

    # menghitung confidence score rata-rata
    avg_probability = float(df_result['Confidence Score'].mean())
    avg_probability = round(avg_probability * 100, 2)

    # Menyimpan file output
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

    return redirect(url_for('summary'))

# Summary and download options
@app.route('/summary', methods=['GET'])
def summary():
    if 'preprocessed_data' not in session:
        return redirect(url_for('index'))
    
    # Mengambil data dari session
    df = pd.DataFrame(session['preprocessed_data'])
    
    # Menganalisis sentimen
    df_result = analyze_sentiment(df)
    sentiment_counts = df_result['sentiment'].value_counts()
    sentiment_counts_dict = sentiment_counts.to_dict()

    # Membuat grafik
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#16A34A', '#2563EB', '#DC2626'])
    ax.axis('equal')

    img_stream = BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)   
    image_data = base64.b64encode(img_stream.getvalue()).decode('utf-8')

    # Menghitung rata-rata confidence score analisis
    avg_probability = float(df_result['Confidence Score'].mean())
    avg_probability = round(avg_probability * 100, 2)

    columns = df_result.columns.tolist()
    values = df_result.values.tolist()

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

    if 'content' not in df.columns:
        return f"Kolom content tidak ditemukan!"

    df_result = analyze_sentiment(df)

    # Generate sentiment graph and PDF
    fig, sentiment_counts = create_sentiment_graph(df_result)
    avg_probability = calculate_average_probability(df_result)

    # Create PDF in memory (without saving it to disk)
    pdf_buffer = BytesIO()
    save_to_pdf(df_result, fig, avg_probability, pdf_buffer=pdf_buffer) 

    # Set buffer position to the beginning
    pdf_buffer.seek(0)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Tentukan path untuk PDF dengan nama file unik
    pdf_filename = f"sentiment_analysis_report_{timestamp}.pdf"

    return send_file(pdf_buffer, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')

@app.route('/download_csv')
def download_csv():
    if 'preprocessed_data' not in session:
        return redirect(url_for('index'))
    
    df = pd.DataFrame(session['preprocessed_data'])
    df_result = analyze_sentiment(df)

    # Membuat file CSV dalam memori menggunakan BytesIO
    csv_buffer = BytesIO()
    df_result.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Mengatur posisi cursor ke awal buffer agar bisa dibaca

    # Mengatur nama file CSV dengan timestamp untuk keunikan
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sentiment_analysis_output_{timestamp}.csv"

    # Mengirim file CSV langsung ke pengguna tanpa menyimpannya ke server
    return send_file(csv_buffer,
                     as_attachment=True,
                     download_name=csv_filename,
                     mimetype='text/csv')

if __name__ == '__main__':
    # Cek apakah folder untuk CSV sudah ada, jika belum buat folder tersebut
    if not os.path.exists(app.config['CSV_OUTPUT_FOLDER']):
        os.makedirs(app.config['CSV_OUTPUT_FOLDER'])
    
    # Cek apakah folder untuk PDF sudah ada, jika belum buat folder tersebut
    if not os.path.exists(app.config['PDF_REPORT_FOLDER']):
        os.makedirs(app.config['PDF_REPORT_FOLDER'])
    
    # Jalankan aplikasi Flask
    app.run(debug=True)
