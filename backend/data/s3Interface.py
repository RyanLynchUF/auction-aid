import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from utils.logger import get_logger

logger = get_logger(__name__)

class S3Uploader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name='us-east-1'):
        self.s3 = boto3.client('s3',
                               aws_access_key_id=aws_access_key_id,
                               aws_secret_access_key=aws_secret_access_key,
                               region_name=region_name)
        self.bucket_name = bucket_name

    def upload_json_to_s3(self, json:str, object_key:str):
        """
        Upload a JSON to specific S3 location.
        
        Parameters:
        -----------
        json : str (JSON format)
            JSON data to be upload to S3 location
        object_key : str
            Key to use for the JSON data in S3.
        """
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=json)
            logger.info(f'Successfully uploaded {object_key} to {self.bucket_name}/{object_key}')
        except NoCredentialsError:
            logger.error('Credentials not available')
        except PartialCredentialsError:
            print("Incomplete credentials provided.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

class S3Reader:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name='us-east-1'):
        self.s3 = boto3.client('s3',
                               aws_access_key_id=aws_access_key_id,
                               aws_secret_access_key=aws_secret_access_key,
                               region_name=region_name)
        self.bucket_name = bucket_name

    def read_from_s3(self, object_key):
        """
        Read data based on object_key from the bucker for the S3Reader.
        
        Parameters:
        -----------
        object_key : str
            Key to read from S3.
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f'Successfully read {object_key} from {self.bucket_name}/{object_key}')

            return response
        except FileNotFoundError:
            logger.error(f'The file {object_key} was not found')
        except NoCredentialsError:
            logger.error('Credentials not available')

