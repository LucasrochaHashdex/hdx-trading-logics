import boto3
import json
import base64
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BigQuery client with project ID from environment
project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID', 'hdx-data-platform')
client = bigquery.Client(project_id)

# Asset lists
list_thematics = ["DEFI11", "WEB311", "META11", "FOMO11"]
list_nasdaq = ["HASH11", "SOLH11", "BITH11", "ETHE11", "XRPH11"]

class TradingAPIManager:
    def __init__(self, secrets_arn: str, region_name: str = None):
        """
        Initialize the Trading API Manager
        
        Args:
            secrets_arn (str): ARN for AWS Secrets
            region_name (str): AWS region name (uses environment variable if not provided)
        """
        self.secrets_arn = secrets_arn
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.secrets = self._load_secrets()
        
        # API endpoints from environment variables
        self.inoa_base_url = os.getenv('INOA_BASE_URL', 'http://10.10.5.8/')
        self.posttrades_base_url = os.getenv('POSTTRADES_BASE_URL', 'https://api.postrade.btgpactual.com')

    def _get_secret(self, secret_name: str) -> str:
        """
        Get secret from AWS Secrets Manager
        
        Args:
            secret_name (str): Name of the secret
            
        Returns:
            str: Secret value
        """
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=self.region_name
        )

        secret_response = client.get_secret_value(SecretId=secret_name)

        if 'SecretString' in secret_response:
            return secret_response['SecretString']
        else:
            return base64.b64decode(secret_response['SecretBinary'])

    def _load_secrets(self) -> Dict[str, str]:
        """
        Load all secrets from AWS
        
        Returns:
            Dict[str, str]: Dictionary of secrets
        """
        secret_str = self._get_secret(self.secrets_arn)
        return json.loads(secret_str)


    def get_post_trades(self, query: str) -> Dict[str, Any]:
        """
        Get data from Post Trades API
        
        Args:
            query (str): API query path
            
        Returns:
            dict: API response
        """
        url = f'{self.posttrades_base_url}{query}'
        headers = {
            'Authorization': f'Bearer {self.secrets["POSTTRADES_API_PASSWORD"]}',
            'Accept': 'application/json',
        }

        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Request failed: Code {response.status_code}, "
                f"Message: {response.text}"
            )

    def get_consolidated_trades(self) -> pd.DataFrame:
        """
        Get and process consolidated trades
        
        Returns:
            pd.DataFrame: Processed trades data
        """
        # Get raw trades data
        trades_data = self.get_post_trades('/allocation/consolidated')
        trades_df = pd.DataFrame(trades_data['consolidatedAllocations'])

        # Process trades DataFrame
        trades_df = trades_df[[
            'accountCode', 'accountAlias', 'side', 'symbol', 
            'qty', 'avgPrice'
        ]]
        
        # Rename columns
        trades_df.columns = [
            'CONTA', 'CONTA_NOME', 'SIDE', 'ATIVO', 
            'QTY. AVAILABLE', 'AVG_PRICE'
        ]

        # Transform data
        trades_df['SIDE'] = trades_df['SIDE'].map({'Buy': 'C', 'Sell': 'V'})
        trades_df = trades_df[~trades_df['ATIVO'].str.endswith('H')]
        trades_df['Financeiro'] = trades_df['QTY. AVAILABLE'] * trades_df['AVG_PRICE']

        return trades_df