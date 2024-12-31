import os
import boto3

def download_s3_folder(bucket_name, prefix, local_dir):
    # Initialize a session using boto3
    s3 = boto3.client('s3')
    
    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Get a list of all objects in the specified bucket/prefix
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                
                # Calculate the relative path to maintain folder structure
                relative_path = os.path.relpath(file_key, prefix)
                local_file_path = os.path.join(local_dir, relative_path)
                
                # Ensure the directory exists before writing the file
                local_dir_path = os.path.dirname(local_file_path)
                if not os.path.exists(local_dir_path):
                    os.makedirs(local_dir_path)
                
                # Download the file
                s3.download_file(bucket_name, file_key, local_file_path)
                print(f"Downloaded {file_key} to {local_file_path}")

if __name__ == "__main__":
    # Define the S3 bucket name
    bucket_name = 'oedi-data-lake'
    
    # Define the folder prefix (the folder inside the bucket)
    prefix = 'SMART-DS/v1.0/2017/AUS/P1R/scenarios/solar_high_batteries_high_timeseries/opendss'
    
    # Define the local directory to save the files
    local_dir = 'path/to/local/directory'
    
    # Call the function to download the folder and its contents
    download_s3_folder(bucket_name, prefix, local_dir)