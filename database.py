from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from credentials import psql_credentials

DATABASE_URL = f'postgresql+psycopg2://{psql_credentials["username"]}:{psql_credentials["password"]}@{psql_credentials["host"]}:{psql_credentials["port"]}/securities_master'

engine = create_engine(DATABASE_URL)

Base = declarative_base()

local_session = sessionmaker(bind=engine, expire_on_commit=False)
