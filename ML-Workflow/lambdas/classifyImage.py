import json
import boto3
import base64

# Fill this in with the name of your deployed model
ENDPOINT_NAME = "image-classification-2023-04-19-11-00-33-457"

def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a boto3 client for invoking SageMaker endpoint
    client = boto3.client('sagemaker-runtime')

    # Invoke the endpoint with the image data
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='image/png',
        Body=image
    )
    
    # Decode the response from SageMaker
    inferences = response['Body'].read().decode()

    # We return the data back to the Step Function    
    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
