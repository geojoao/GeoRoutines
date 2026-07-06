"""
Entrada/saída via Azure Blob Storage.

Segue o mesmo padrão da rotina ``modis-evi-hexagono``: usa a variável de
ambiente ``AZURE_STORAGE_CONNECTION_STRING`` e os containers
``planetary-routines-input`` (entrada) e ``planetary-routines-output`` (saída).
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config


def _service():
    from azure.storage.blob import BlobServiceClient

    connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    return BlobServiceClient.from_connection_string(connect_str)


def download(blob_name: str, dest: str | Path, container: str) -> bool:
    """
    Baixa ``blob_name`` do ``container`` para ``dest``. Retorna ``True`` se o
    blob existia (e foi baixado), ``False`` caso contrário.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = _service().get_blob_client(container=container, blob=blob_name)
    if not client.exists():
        print(f"[blob] {container}/{blob_name} não existe")
        return False
    with open(dest, "wb") as f:
        f.write(client.download_blob().readall())
    print(f"[blob] baixado {container}/{blob_name} -> {dest}")
    return True


def upload(local_path: str | Path, blob_name: str, container: str,
           overwrite: bool = True) -> None:
    """Envia ``local_path`` para ``container/blob_name`` (sobrescreve)."""
    local_path = Path(local_path)
    client = _service().get_blob_client(container=container, blob=blob_name)
    if overwrite and client.exists():
        client.delete_blob()
    with open(local_path, "rb") as data:
        client.upload_blob(data, overwrite=overwrite)
    print(f"[blob] enviado {local_path} -> {container}/{blob_name}")


def list_blobs(prefix: str, container: str) -> list[str]:
    """Lista nomes de blobs sob ``prefix`` no ``container``."""
    cc = _service().get_container_client(container)
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


def delete(blob_name: str, container: str) -> None:
    client = _service().get_blob_client(container=container, blob=blob_name)
    if client.exists():
        client.delete_blob()


# ---------------------------------------------------------------------------
# Helpers com os caminhos-padrão da rotina
# ---------------------------------------------------------------------------
def output_blob_name(*parts: str) -> str:
    return "/".join([config.BLOB_OUTPUT_PREFIX, *parts])


def download_input(blob_name: str, dest: str | Path) -> bool:
    return download(blob_name, dest, config.BLOB_INPUT_CONTAINER)


def download_output(blob_name: str, dest: str | Path) -> bool:
    return download(blob_name, dest, config.BLOB_OUTPUT_CONTAINER)


def upload_output(local_path: str | Path, blob_name: str) -> None:
    upload(local_path, blob_name, config.BLOB_OUTPUT_CONTAINER)
