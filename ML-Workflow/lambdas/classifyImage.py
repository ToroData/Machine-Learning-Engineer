import json
import boto3
import base64

ENDPOINT_NAME = "image-classification-2023-04-20-15-20-25-339"

def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a boto3 client for invoking SageMaker endpoint
    client = boto3.client('sagemaker-runtime')
    
    # Serialize the image data in the correct format for SageMaker
    image_content = bytearray(image)
    content_type = 'image/png'
    accept = 'application/json'

    # Invoke the endpoint with the image data
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='image/png',
        Accept=accept,
        Body=image_content
    )
    
    # Decode the response from SageMaker
    response_body = json.loads(response['Body'].read().decode())

    # We return the data back to the Step Function    
    event["body"]["inferences"] = response_body
    
    return {
        "statusCode": 200,
        "body": {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['body']['inferences'],
        }
    }
