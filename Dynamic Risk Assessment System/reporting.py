"""
Module for generating model performance reports and conducting diagnostics.

This module provides functions for generating machine learning model performance reports, 
including generating confusion matrices, creating detailed PDF reports, and logging model
performance diagnostics over time.

Functions:
    - generate_confusion_matrix(save_path=None) -> pd.DataFrame:
        Generate a confusion matrix and optionally save it as a PNG file.

    - download_logo(url: str, temp_dir: str) -> str:
        Download a logo from the provided URL and save it to a temporary directory.

    - add_logo_to_header(canvas, doc, logo_path: str) -> None:
        Add a logo to the header of a PDF document.

    - create_header(logo_path: str):
        Create a custom header function with a specified logo path.

    - read_times_from_directories(parent_dir: str) -> list:
        Read time records from directories and return them as a list.

    - create_times_graph(time_records: list, graph_path: str) -> None:
        Create a bar plot to visualize data ingestion and model training times.

    - generate_pdf_report(confusion_matrix: pd.DataFrame) -> str:
        Generate a PDF report with a confusion matrix plot.

    - score_model():
        Execute the model scoring process, generate a confusion matrix, and create a PDF report.

Author: Ricard Santiago Raigada García
Date: January, 2024
"""
import json
import os
import sys
import logging
import requests
from shutil import rmtree
from tempfile import mkdtemp
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import PageBreak
from diagnostics import model_predictions, dataframe_summary
from training import split_data


# Load config.json and get path variables
with open('config.json', 'r') as f:
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
    confusion_matrix = pd.crosstab(
        y_true,
        y_pred,
        rownames=['Actual'],
        colnames=['Predicted'])

    # Generate a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar_kws={
                'shrink': .5})  # fmt='g' to avoid scientific notation
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)

    # Optionally save the plot as a PNG file
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()

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


def add_logo_to_header(canvas, doc, logo_path: str) -> None:
    """
    Add a logo to the header of a PDF document.

    Args:
        canvas (reportlab.pdfgen.canvas.Canvas): The canvas of the PDF document.
        doc (reportlab.platypus.doctemplate.BaseDocTemplate): The PDF document itself.
        logo_path (str): The path to the logo image file.

    Returns:
        None
    """
    canvas.drawImage(
        logo_path,
        doc.width +
        doc.rightMargin -
        1 *
        inch,
        A4[1] -
        1 *
        inch,
        width=1 *
        inch,
        height=1 *
        inch,
        mask='auto')


def create_header(logo_path: str):
    """
    Create a header function with a specified logo path.

    Args:
        logo_path (str): The path to the logo image file.

    Returns:
        function: A header function that can be used in a PDF document template.
    """
    def header(canvas, doc):
        add_logo_to_header(canvas, doc, logo_path)
    return header


def read_ingested_files_info(file_path: str) -> list:
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


def read_latest_f1_score(file_path: str) -> str:
    """
    Reads the txt and extracts the F1 score.

    Args:
        file_path (str): The path to txt.

    Returns:
        str: The F1 score read from the file.
    """
    with open(file_path, 'r') as file:
        f1_score = file.read().strip()
    return f1_score


def read_times_from_directories(parent_dir: str) -> list:
    """
    Read time records from directories and return them as a list.

    Args:
        parent_dir (str): The parent directory containing subdirectories with time records.

    Returns:
        list: A list of time records, where each record is a tuple containing data ingestion time (str)
        and model training time (str) for a specific run.
    """
    time_records = []
    for dir_name in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(dir_path):
            ingestion_time_path = os.path.join(
                dir_path, f"data_ingestion_time_{dir_name}.txt")
            training_time_path = os.path.join(
                dir_path, f"model_training_time_{dir_name}.txt")

            with open(ingestion_time_path, 'r') as file:
                ingestion_time = file.read().split(
                    ":")[1].strip().split(" ")[0]

            with open(training_time_path, 'r') as file:
                training_time = file.read().split(":")[1].strip().split(" ")[0]

            time_records.append((ingestion_time, training_time))
    return time_records


def create_times_graph(time_records: list, graph_path: str) -> None:
    """
    Create a bar plot to visualize data ingestion and model training times.

    Args:
        time_records (list): A list of time records where each record is a list
            containing data ingestion time (in seconds) and model training time (in seconds).
        graph_path (str): The path to save the generated graph as an image.

    Returns:
        None
    """
    # Convert the time records to numpy arrays for easy plotting
    ingestion_times = np.array([float(record[0]) for record in time_records])
    training_times = np.array([float(record[1]) for record in time_records])

    # Create a range for the x-axis
    n_groups = len(ingestion_times)
    index = np.arange(n_groups)

    # Create the bar plot
    fig, ax = plt.subplots()

    if ingestion_times.min() > 0:
        ax.set_yscale('log')

    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, ingestion_times, bar_width,
                    alpha=opacity, color='b',
                    label='Data Ingestion Time')

    rects2 = ax.bar(index + bar_width, training_times, bar_width,
                    alpha=opacity, color='g',
                    label='Model Training Time')

    ax.set_xlabel('Runs')
    ax.set_ylabel('Seconds')
    ax.set_title('Data Ingestion and Model Training Times by Run')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([str(i + 1) for i in range(n_groups)])
    ax.legend()

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()


def generate_pdf_report(confusion_matrix: pd.DataFrame) -> str:
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
    doc = SimpleDocTemplate(
        report_filename,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=120,
        bottomMargin=72)
    story = [Spacer(1, -0.5 * inch)]

    # Add a title
    story.append(Paragraph("Model Performance Report", styles['Title']))
    story.append(Spacer(1, 0.25 * inch))
    story.append(PageBreak())

    # Add description
    story.append(Paragraph("Description of data", styles['Heading1']))
    story.append(Spacer(1, 0.25 * inch))
    dataset_description = """
    This report incorporates a detailed analysis of a key data set used to predict the risk of corporate customer defection.\
    The data set excludes direct corporate identifiers, focusing predictive modeling on significant operational variables. These \
    variables include the monthly and annual activity of the client, which reflects the level of recent and sustained \
    interaction, as well as the size of the client, represented by the number of employees. These metrics are essential\
    to feed the machine learning model, whose objective is to anticipate the termination of contracts. By accurately forecasting churn risk,\
    account managers can strategically prioritize customer retention. The dynamic approach to this analysis ensures that the model stays\
    up-to-date in the face of changing market trends, allowing the company to minimize customer churn and maximize revenue retention.
    """
    story.append(Paragraph(dataset_description, styles['Normal']))
    story.append(Spacer(1, 0.25 * inch))
    story.append(PageBreak())

    # Read and add ingestion files information
    ingestion_info_path = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    ingestion_info = read_ingested_files_info(ingestion_info_path)
    styles = getSampleStyleSheet()
    story.append(Paragraph("Ingested Files Information:", styles['Heading2']))
    for info in ingestion_info:
        story.append(Paragraph(info, styles['Normal']))
    story.append(Spacer(1, 0.25 * inch))

    # Add summary statistics
    story.append(Paragraph("Dataframe statistics:", styles['Heading2']))
    summary_stats = dataframe_summary()
    header = ['Statistic', 'Mean', 'Median', 'StdDev']
    data = [header] + summary_stats
    t = Table(data, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)

    # Read and add F1 score information
    latest_score_path = os.path.join(model_path, "latestscore.txt")
    f1_score = read_latest_f1_score(latest_score_path)
    f1_explanation = """The F1 score is the harmonic mean of precision and recall, providing a balance between them.
                        It is a good measure to use if you need to seek a balance between precision and recall and there
                        is an uneven class distribution (large number of actual negatives)."""
    styles = getSampleStyleSheet()
    story.append(
        Paragraph(
            "F1 Score Interpretation and Current Value:",
            styles['Heading2']))
    story.append(Paragraph(f1_explanation, styles['Normal']))
    story.append(
        Paragraph(
            f"The actual F1 score is {f1_score}",
            styles['Normal']))
    story.append(Spacer(1, 0.25 * inch))
    story.append(PageBreak())

    # Add the confusion matrix plot
    story.append(Paragraph("Confusion Matrix:", styles['Heading2']))
    cm_img = Image("practicemodels/confusionmatrix.png", 4 * inch, 3 * inch)
    cm_img.hAlign = 'CENTER'
    story.append(cm_img)
    story.append(Spacer(1, 0.25 * inch))

    # Read and add times information from old diagnostics
    story.append(PageBreak())
    old_diagnostics_dir = "olddiagnostics"
    times_records = read_times_from_directories(old_diagnostics_dir)
    time_header = [
        'Data Ingestion Time (seconds)',
        'Model Training Time (seconds)']
    time_data = [time_header] + times_records

    story.append(Paragraph("Previous Run Times:", styles['Heading2']))
    time_table = Table(time_data)
    time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(time_table)
    story.append(Spacer(1, 0.25 * inch))

    # Create and add the times graph
    graph_path = os.path.join(temp_dir, 'times_graph.png')
    create_times_graph(times_records, graph_path)
    story.append(
        Paragraph(
            "Data Ingestion and Model Training Times Graph:",
            styles['Heading2']))
    times_graph_img = Image(graph_path, 6 * inch, 4 * inch)
    times_graph_img.hAlign = 'CENTER'
    story.append(times_graph_img)
    story.append(Spacer(1, 0.25 * inch))

    # Use the header function with the logo_path
    doc.build(story, onFirstPage=create_header(logo_path),
              onLaterPages=create_header(logo_path))

    # Clean up the temporary directory
    rmtree(temp_dir)

    return report_filename


def score_model():
    """
    Execute the model scoring process, generate a confusion matrix, and create a PDF report.

    This function orchestrates the scoring of the machine learning model by generating a confusion matrix
    for the model predictions against the actual values. It then proceeds to call the function to generate
    a PDF report which includes the confusion matrix, model performance metrics, and additional insights.

    The confusion matrix is saved as a PNG image at the specified path within the model's directory.
    The PDF report is created with a comprehensive view of the model's performance and additional data analysis,
    which is then saved to the file system.

    Args:
        None

    Returns:
        None

    The function does not take any arguments and does not return any values. Instead, it performs file I/O operations,
    writing the confusion matrix image and PDF report to the disk. It relies on global paths defined in the configuration
    to locate the necessary files and directories for its operation.
    """
    confusion_matrix = generate_confusion_matrix(
        os.path.join(model_path, 'confusionmatrix.png'))
    pdf_report = generate_pdf_report(confusion_matrix)


if __name__ == '__main__':
    score_model()
