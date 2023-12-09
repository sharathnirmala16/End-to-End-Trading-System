from typing import Dict, List, Tuple
from datetime import datetime
from celery_config import celery
from credentials import psql_credentials
from Commons.enums import *
from Commons.common_types import DownloadRequest
from Commons.common_tasks import CommonTasks
from Vendors.api_manager import APIManager

import pandas as pd
import sqlalchemy


def create_engine():
    try:
        url = f"postgresql+psycopg2://{psql_credentials['username']}:{psql_credentials['password']}@{psql_credentials['host']}:{psql_credentials['port']}/securities_master"
        engine = sqlalchemy.create_engine(url, isolation_level="AUTOCOMMIT")
        return engine
    except Exception as e:
        raise e


engine = create_engine()


class Tasks:
    pass


if __name__ == "__main__":
    celery.start()
