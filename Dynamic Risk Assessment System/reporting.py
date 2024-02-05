import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import logging
import requests
from shutil import rmtree
from tempfile import mkdtemp
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from diagnostics import model_predictions, dataframe_summary
from training import split_data

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def generate_confusion_matrix(save_path=None) -> pd.DataFrame:
    """
    Generate a confusion matrix plot and optionally save it as a PNG file.

    Args:
        save_path (str, optional): File path to save the confusion matrix plot as a PNG. Defaults to None.

    Returns:
        pd.DataFrame: Confusion matrix.
    """
    # Loading and preparing testdata.csv
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    # Predicting test data
    logging.info("Predicting test data")
    y_pred = model_predictions()
    _, _, y_true, _ = split_data(test_df)
    
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    
    # Generate a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar_kws={'shrink': .5})  # fmt='g' to avoid scientific notation
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    
    # Optionally save the plot as a PNG file
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    # plt.show()
    
    return confusion_matrix


def download_logo(url: str, temp_dir: str) -> str:
    """
    Download a logo from the provided URL and save it to a temporary directory.

    Args:
        url: URL to the logo image
        temp_dir: Path to the temporary directory

    Returns:
        Path to the downloaded logo image
    """
    response = requests.get(url)
    if response.status_code == 200:
        logo_path = os.path.join(temp_dir, 'logo.png')
        with open(logo_path, 'wb') as f:
            f.write(response.content)
        return logo_path
    else:
        print(f"Error downloading the logo: {response.status_code}")
        return None

# Header function to be used in the doc template for adding the logo
def add_logo_to_header(canvas, doc, logo_path):
    canvas.drawImage(logo_path, doc.width + doc.rightMargin - 1*inch, A4[1] - 1*inch, width=1*inch, height=1*inch, mask='auto')

# Wrapper function to pass 'logo_path' to the header function
def create_header(logo_path):
    def header(canvas, doc):
        add_logo_to_header(canvas, doc, logo_path)
    return header

def read_ingested_files_info(file_path):
    """
    Reads the 'ingestedfiles.txt' and formats the information for inclusion in the PDF.

    Args:
        file_path (str): The path to 'ingestedfiles.txt'.

    Returns:
        list: Formatted ingestion information for each file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process and format each line
    formatted_lines = []
    for line in lines:
        parts = line.split('\t')  # Assuming the file is tab-delimited
        if len(parts) == 2:
            formatted_lines.append(f"{parts[0].strip()} - {parts[1].strip()}")

    return formatted_lines

def generate_pdf_report(confusion_matrix):
    """
    Generate a PDF report with a confusion matrix plot.

    Args:
        confusion_matrix: Confusion matrix as a pandas DataFrame

    Returns:
        The report filename
    """
    # Get current timestamp for the report filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    report_filename = f'report_{timestamp}.pdf'
    
    # Create a temporary directory
    temp_dir = mkdtemp()
    
    # Download the logo
    logo_url = "http://thedatascientist.digital/img/logo.png"
    logo_path = download_logo(logo_url, temp_dir)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_filename, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=120, bottomMargin=72)
    story = [Spacer(1, -0.5*inch)]

    # Add a title
    story.append(Paragraph("Model Performance Report", styles['Title']))
    story.append(Spacer(1, 0.25*inch))
    
    # Read and add ingestion files information
    ingestion_info_path = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    ingestion_info = read_ingested_files_info(ingestion_info_path)
    styles = getSampleStyleSheet()
    story.append(Paragraph("Ingested Files Information:", styles['Heading2']))
    for info in ingestion_info:
        story.append(Paragraph(info, styles['Normal']))
    story.append(Spacer(1, 0.25*inch))

    # Add the confusion matrix plot
    story.append(Paragraph("Confusion Matrix:", styles['Heading2']))
    cm_img = Image("practicemodels/confusionmatrix.png", 4*inch, 3*inch)
    cm_img.hAlign = 'CENTER'
    story.append(cm_img)
    story.append(Spacer(1, 0.25*inch))

    # Add summary statistics
    summary_stats = dataframe_summary()
    header = ['Statistic', 'Mean', 'Median', 'StdDev']
    data = [header] + summary_stats
    t = Table(data, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(t)

    # Use the header function with the logo_path
    doc.build(story, onFirstPage=create_header(logo_path), onLaterPages=create_header(logo_path))

    # Clean up the temporary directory
    rmtree(temp_dir)

    return report_filename


def score_model():
    confusion_matrix = generate_confusion_matrix(os.path.join(model_path, 'confusionmatrix.png'))
    pdf_report = generate_pdf_report(confusion_matrix)


if __name__ == '__main__':
    score_model()
