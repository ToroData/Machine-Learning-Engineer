import zipfile
import os
import sys
from tqdm import tqdm
import shutil
import boto3


def unzipper(filename):
    """
    Extracts the contents of a ZIP file to a folder named 'tmp' in the current working directory.
    If the folder 'tmp' already exists, the function will overwrite its contents.
    
    Args:
    -----
        filename (str): The name of the ZIP file to extract.
    
    Returns:
    --------
        None
    
    Raises:
    -------
        Exception: If the ZIP file cannot be opened or extracted.
    """
    try:
        with zipfile.ZipFile(filename) as zf:
            os.makedirs('tmp', exist_ok=True)
            # Iterate over the members of the ZIP file and extract them to 'tmp'
            for member in tqdm(zf.infolist(), desc='Extracting '):
                try:
                    zf.extract(member, os.path.join(os.getcwd(), 'tmp'))
                except zipfile.error as e:
                    print('Failed'+ str(e))    
        print('Extracted')
    except Exception as e:
        print('Failed'+ str(e))
        
        
s3 = boto3.client('s3')

def upload_to_s3(bucket_name, local_path):
    """
    Uploads files from a local directory to an Amazon S3 bucket.
    If the bucket does not exist, it is created.
    
    Args:
    ----
        bucket_name (str): The name of the Amazon S3 bucket to upload files to.
        local_path (str): The local path of the directory containing the files to upload.
    
    Returns:
    --------
        None
    
    Raises:
    -------
        Exception: If there is an error creating or uploading to the S3 bucket.
    """
    # Check if the bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except:
        # If the bucket does not exist, create it
        s3.create_bucket(Bucket=bucket_name)
        print('[INFO] Bucket created')
    
    # Upload the files to S3
    for root, dirs, files in os.walk(local_path):
        for file in tqdm(files, desc='Uploading files'):
            local_file = os.path.join(root, file)
            s3_file = os.path.relpath(local_file, local_path)
            s3.upload_file(local_file, bucket_name, s3_file)
    
    # Remove the temporary directory and all its contents
    try:
        shutil.rmtree(local_path)
        print('[INFO] Temporary directory deleted')
    except Exception as e:
        print('[ERROR] Failed to delete temporary directory: {}'.format(str(e)))

    print('[INFO] Uploaded successfully')
    
    


if __name__=='__main__':
    filename = sys.argv[1]
    bucket_name = sys.argv[2]
    local_path = os.path.join(os.getcwd(), 'tmp')
    unzipper(filename)
    upload_to_s3(
        bucket_name,
        local_path
    )

