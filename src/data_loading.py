from pathlib import Path
import ssl
import pandas as pd
from sklearn.datasets import fetch_california_housing
import urllib.request

TELCO_IBM_RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def _download_if_missing(url: str, dest_path: Path) -> Path:
    """
    Download the data from the given URL to the destination path.

    Args:
        url (str): the URL to download the data from
        dest_path (Path): the destination path to save the data
    
    Returns:
        Path: the destination path
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        # Use certifi's CA bundle so SSL verification works (e.g. on macOS with Python.org installs)
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(url) as response:
            dest_path.write_bytes(response.read())
    return dest_path

def load_telco(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Loads the IBM Telco churn dataset.
    If missing locally, downloads it from a public IBM GitHub raw URL.
    
    Args:
        data_dir (str | Path): the directory to save the data

    Returns:
        pd.DataFrame: the loaded dataset
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "telco_customer_churn.csv"
    _download_if_missing(TELCO_IBM_RAW_URL, csv_path)
    df = pd.read_csv(csv_path)
    return df

def load_housing(as_frame: bool = True) -> pd.DataFrame:
    """    
    Loads the California Housing dataset using scikit-learn's fetch function.
    scikit-learn will download/cache automatically.

    Args:
        as_frame (bool): whether to return the data as a pandas DataFrame

    Returns:
        pd.DataFrame: the loaded dataset
    """ 
    bunch = fetch_california_housing(as_frame=as_frame)
    return bunch.frame 