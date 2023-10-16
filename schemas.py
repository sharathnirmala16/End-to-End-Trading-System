from typing import List, Dict
from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    name: str
    email: str
    password: str


class TokenSchema(BaseModel):
    access_token: str


class ChangePassword(BaseModel):
    username: str
    old_password: str
    new_password: str


class RequestDetails(BaseModel):
    username: str
    password: str


class PricesRequest(BaseModel):
    index: str = ""
    tickers: List[str] = []
    interval: int
    start_datetime: str
    end_datetime: str
    vendor: str
    exchange: str
    instrument: str
    vendor_login_credentials: Dict[str, str] = {}
    cache_data: bool = False
