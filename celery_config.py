from credentials import rabbitmq_credentials as rc
from credentials import psql_credentials as pc
from celery import Celery

CELERY_BROKER_URL = (
    f"pyamqp://{rc['username']}:{rc['password']}@{rc['host']}/{rc['vhost']}"
)
CELERY_BACKEND_URL = (
    f"db+postgresql://{pc['username']}:{pc['password']}@{pc['host']}/securities_master"
)

celery = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_BACKEND_URL)
