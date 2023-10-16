from sqlalchemy import Column, String, DateTime, Boolean
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"
    username = Column(String(32), nullable=False, primary_key=True)
    name = Column(String(64), nullable=False)
    email = Column(String(64), nullable=False)
    password = Column(String(256), unique=True, nullable=False)


class TokenTable(Base):
    __tablename__ = "tokens"
    username = Column(String(32))
    access_token = Column(String(512), primary_key=True)
    status = Column(Boolean)
    created_datetime = Column(DateTime, default=datetime.now)
