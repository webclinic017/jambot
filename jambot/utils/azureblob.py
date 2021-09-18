"""
Azure Blob storage

More file operation examples:
https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/storage/azure-storage-blob/samples/blob_samples_directory_interface.py
"""

from pathlib import Path
from typing import Union

from azure.storage.blob import (BlobClient, BlobServiceClient,  # noqa
                                ContainerClient)

from jambot import functions as f
from jambot import getlog
from jambot.utils.secrets import SecretsManager

log = getlog(__name__)


class BlobStorage():
    def __init__(self, container: Union[str, Path] = 'jambot-app') -> None:
        """

        Parameters
        ----------
        container : str, optional
            container to use by default, by default 'jambot-app'
        """
        creds = SecretsManager('azure_blob.yaml').load
        client = BlobServiceClient.from_connection_string(creds['connection_string'])

        # pass in full dir, but only use name
        _p_local = None
        if isinstance(container, Path):
            _p_local = container
            container = container.name

        f.set_self(vars())

    @property
    def p_local(self) -> Path:
        """Local directory to upload/download to/from"""
        if self._p_local is None:
            raise RuntimeError('self.p_local not set!')

        return self._p_local

    def get_container(self, container: Union[str, ContainerClient] = None) -> ContainerClient:
        """Get container object

        Parameters
        ----------
        container : Union[str, ContainerClient]
            container/container name

        Returns
        -------
        ContainerClient
        """

        # container already init
        if isinstance(container, ContainerClient):
            return container

        container = container or self.container
        return self.client.get_container_client(container)

    def clear_container(self, container: Union[str, ContainerClient] = None) -> None:
        """Delete all files in container

        Parameters
        ----------
        container : Union[str, ContainerClient]
            container name
        """
        container = self.get_container(container)
        blob_list = [b.name for b in container.list_blobs()]

        # Delete blobs
        container.delete_blobs(*blob_list)

    def upload_dir(
            self,
            p: Path = None,
            container: Union[str, ContainerClient] = None,
            mirror: bool = True) -> None:
        """Upload entire dir files to container

        Parameters
        ----------
        p : Path
            dir to upload, default self.p_local
        container : Union[str, ContainerClient], optional
        mirror : bool, optional
            if true, delete all contents from container first
        """
        if p is None:
            p = self.p_local

        self._validate_dir(p)
        container = self.get_container(container)

        if mirror:
            self.clear_container(container)

        i = 0
        for _p in p.iterdir():
            if not _p.is_dir():
                self.upload_file(p=_p, container=container)
                i += 1

        log.info(f'Uploaded [{i}] file(s) to container "{container.container_name}"')

    def download_dir(
            self,
            p: Path = None,
            container: Union[str, ContainerClient] = None,
            mirror: bool = True) -> None:
        """Download entire container to local dir

        Parameters
        ----------
        p : Path
            dir to download to
        container : Union[str, ContainerClient], optional
        mirror : bool, optional
            if true, clear local dir first, by default True
        """
        if p is None:
            p = self.p_local

        self._validate_dir(p)
        container = self.get_container(container)

        if mirror:
            for _p in p.iterdir():
                _p.unlink()

        # blob here is BlobProperties
        i = 0
        for blob in container.list_blobs():
            self.download_file(p=p / blob.name, container=container)
            i += 1

        log.info(f'Downloaded [{i}] file(s) from container "{container.container_name}"')

    def download_file(
            self,
            p: Path,
            container: Union[str, ContainerClient] = None) -> Path:
        """Download file from container and save to local file

        Parameters
        ----------
        p : Path
            path to save to, p.name will be used to find file in blob
        container : str, optional
        """
        container = self.get_container(container)
        blob = container.get_blob_client(p.name)

        with open(p, 'wb') as file:
            file.write(blob.download_blob().readall())

        return blob

    def upload_file(self, p: Path, container: str = None) -> None:
        """Save local file to container

        Parameters
        ----------
        p : Path
            Path obj to upload to blob storage
        container : str, optional
            container name, by default self.container
        """
        container = self.get_container(container)

        if not p.exists():
            raise FileNotFoundError(f'Data file: "{p.name}" does not exist.')

        with open(p, 'rb') as file:
            blob = container.upload_blob(name=p.name, data=file)

    def show_containers(self) -> None:
        """Show list of container names"""
        names = [c.name for c in self.client.list_containers()]
        f.pretty_dict(names)

    def show_files(self, container: str = None) -> None:
        """Print list of files in container

        Parameters
        ----------
        container : str, optional
            container to show files in, default self.container
        """
        container = self.get_container(container)
        names = [b.name for b in container.list_blobs()]
        f.pretty_dict(names)

    def create_container(self, name: str) -> None:
        """Wrapper to create container in storage account

        Parameters
        ----------
        name : str
            name of container
        """
        self.client.create_container(name)

    def _validate_dir(self, p: Path) -> None:
        """Check if path is valid directory

        Parameters
        ----------
        p : Path
            path to check
        """
        if not p.is_dir():
            raise ValueError(f'p is not a directory: "{p}"')
