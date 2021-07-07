"""
Azure Blob storage
"""

import os
import uuid

from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient

from jambot.utils.secrets import SecretsManager


class BlobStorage():
    def __init__(self) -> None:
        creds = SecretsManager('azure_blob.yaml').load
        self.client = BlobServiceClient.from_connection_string(creds['connection_string'])
        self.bucket = 'jambot-app'

    def get_file(self, name: str, bucket: str = None):
        if bucket is None:
            bucket = self.bucket

        return

    def save_file(self, name: str, bucket: str = None):
        if bucket is None:
            bucket = self.bucket
