import utils
import models
import asyncio
import schemas
import database
import auth_bearer
import numpy as np
import pandas as pd
import sqlalchemy

from functools import wraps
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, List
from credentials import psql_credentials
from datetime import datetime
from SecuritiesMaster.securities_master import SecuritiesMaster
from celery.result import AsyncResult
from celery_server import Tasks


database.Base.metadata.create_all(database.engine)


def get_session():
    session = database.local_session()
    try:
        yield session
    except:
        session.close()


app = FastAPI(
    title="Sphera API",
    description="The Sphera algorithmic trading API provides the methods to automate your trading.",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

securities_master = SecuritiesMaster(
    psql_credentials["host"],
    psql_credentials["port"],
    psql_credentials["username"],
    psql_credentials["password"],
)


def token_required(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        payload = auth_bearer.JWTBearer.decodeJWT(kwargs["dependencies"])
        username = payload["sub"]
        data = (
            kwargs["session"]
            .query(models.TokenTable)
            .filter_by(
                username=username, access_token=kwargs["dependencies"], status=True
            )
            .first()
        )
        if data:
            return await func(*args, **kwargs)

        else:
            return {"msg": "Token blocked"}

    return wrapper


@app.post("/register", tags=["Authentication"])
async def register_user(
    user: schemas.UserCreate, session: Session = Depends(get_session)
):
    existing_user: models.User | None = (
        session.query(models.User).filter_by(username=user.username).first()
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    new_user = models.User(
        username=user.username,
        name=user.name,
        email=user.email,
        password=utils.get_hashed_password(user.password),
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {"message": f"User {new_user.username} created successfully"}


@app.post("/login", response_model=schemas.TokenSchema, tags=["Authentication"])
async def login(
    request: schemas.RequestDetails, session: Session = Depends(get_session)
) -> Dict:
    user: models.User = (
        session.query(models.User)
        .filter(models.User.username == request.username)
        .first()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username"
        )
    hashed_password = user.password
    if not utils.verify_password(request.password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect password"
        )

    access_token = utils.create_access_token(user.username)

    token_db = models.TokenTable(
        username=user.username, access_token=access_token, status=True
    )

    session.add(token_db)
    session.commit()
    session.refresh(token_db)

    return {"access_token": access_token}


@app.post("/change-password", tags=["Authentication"])
@token_required
async def change_password(
    request: schemas.ChangePassword,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    user: models.User = (
        session.query(models.User)
        .filter(models.User.username == request.username)
        .first()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found"
        )

    if not utils.verify_password(request.old_password, user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid old password"
        )

    encrypted_password = utils.get_hashed_password(request.new_password)
    user.password = encrypted_password
    session.commit()

    return {"message": "Password changed successfully"}


@app.post("/logout", tags=["Authentication"])
@token_required
async def logout(
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    token = dependencies
    payload = auth_bearer.JWTBearer.decodeJWT(token)
    username = payload["sub"]
    token_record = session.query(models.TokenTable).all()
    info = []
    for record in token_record:
        if (datetime.utcnow() - record.created_datetime).days > 1:
            info.append(record.username)
    if info:
        existing_token = (
            session.query(models.TokenTable)
            .where(models.TokenTable.username.in_(info))
            .delete()
        )
        session.commit()

    existing_token = (
        session.query(models.TokenTable)
        .filter(
            models.TokenTable.username == username,
            models.TokenTable.access_token == token,
        )
        .first()
    )
    if existing_token:
        existing_token.status = False
        session.add(existing_token)
        session.commit()
        session.refresh(existing_token)
    return {"message": "Logged out successfully"}


@app.get("/get-details/{username}", tags=["Authentication"])
@token_required
async def get_details(
    username: str,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    user: models.User = (
        session.query(models.User).filter(models.User.username == username).first()
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found"
        )

    return {"name": user.name, "email": user.email}


@app.put("/change-email", tags=["Authentication"])
@token_required
async def change_email(
    username: str,
    new_email: str,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    user: models.User = (
        session.query(models.User).filter(models.User.username == username).first()
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found"
        )

    session.execute(
        sqlalchemy.text(
            f"UPDATE users SET email = '{new_email}' WHERE username = '{username}'"
        )
    )
    session.commit()

    return {"message": "email changed successfully"}


@app.put("/change-name", tags=["Authentication"])
@token_required
async def change_name(
    username: str,
    new_name: str,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    user: models.User = (
        session.query(models.User).filter(models.User.username == username).first()
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found"
        )

    session.execute(
        sqlalchemy.text(
            f"UPDATE users SET name = '{new_name}' WHERE username = '{username}'"
        )
    )
    session.commit()

    return {"message": "name changed successfully"}


# SECURITIES_MASTER
@app.get("/securities-master/get-all-tables", tags=["Securities Master"])
@token_required
async def get_all_tables(
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    try:
        tables: List[str] = securities_master.get_all_tables()
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/securities-master/{table_name}", tags=["Securities Master"])
@token_required
async def get_table(
    table_name: str,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    try:
        if table_name not in securities_master.get_all_tables():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Table not found"
            )

        table = securities_master.get_table(table_name)

        if pd.api.types.is_datetime64_any_dtype(table.index.to_series()):
            table.index = table.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        for column in table.columns:
            if pd.api.types.is_datetime64_any_dtype(table[column]):
                table[column] = table[column].dt.strftime("%Y-%m-%d %H:%M:%S")

        return JSONResponse(content=table.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.delete("/securities-master/{table_name}/delete-table", tags=["Securities Master"])
@token_required
async def delete_table(
    table_name: str,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    if table_name not in securities_master.get_all_tables():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Table not found"
        )
    try:
        securities_master.delete_table(table_name)
        return {"message": "table deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/securities-master/{table_name}/add-row", tags=["Securities Master"])
@token_required
async def add_rows(
    table_name: str,
    row_data: Dict[str, int | float | str | None],
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    if table_name not in securities_master.get_all_tables():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Table not found"
        )
    if (
        list(row_data.keys())
        == securities_master.get_table(table_name).columns.to_list()
    ):
        try:
            securities_master.add_row(table_name, row_data)
            return {"message": "Added a new row successfully"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid rows"
        )


@app.put("/securities-master/{table_name}/edit-row", tags=["Securities Master"])
@token_required
async def edit_row(
    table_name: str,
    old_row_data: Dict[str, int | float | str | None],
    new_row_data: Dict[str, int | float | str | None],
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    if table_name not in securities_master.get_all_tables():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Table not found"
        )
    if set(new_row_data.keys()).issubset(
        set(securities_master.get_table(table_name).columns.to_list())
    ):
        try:
            securities_master.edit_row(table_name, old_row_data, new_row_data)
            return {"message": "Edited a row successfully"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid rows"
        )


@app.delete("/securities-master/{table_name}/delete-row", tags=["Securities Master"])
@token_required
async def delete_row(
    table_name: str,
    row_data: Dict[str, str],
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
) -> JSONResponse:
    if table_name not in securities_master.get_all_tables():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Table not found"
        )
    try:
        securities_master.delete_row(table_name, row_data)
        return {"message": "Row deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# NOTE: Add Schema
@app.post("/securities-master/get-prices", tags=["Securities Master"])
@token_required
async def get_prices(
    prices_request: schemas.PricesRequest,
    session: Session = Depends(get_session),
    dependencies=Depends(auth_bearer.JWTBearer()),
):
    try:
        prices_request.start_datetime = datetime.strptime(
            prices_request.start_datetime, "%Y-%m-%d %H:%M:%S"
        )
        prices_request.end_datetime = datetime.strptime(
            prices_request.end_datetime, "%Y-%m-%d %H:%M:%S"
        )

        data: Dict[str, pd.DataFrame] = securities_master.get_prices(
            index=prices_request.index,
            tickers=prices_request.tickers,
            interval=prices_request.interval,
            start_datetime=prices_request.start_datetime,
            end_datetime=prices_request.end_datetime,
            vendor=prices_request.vendor,
            exchange=prices_request.exchange,
            instrument=prices_request.instrument,
            vendor_login_credentials=prices_request.vendor_login_credentials,
            cache_data=prices_request.cache_data,
        )

        for ticker in data:
            table = data[ticker].copy(deep=True)
            if pd.api.types.is_datetime64_any_dtype(table.index.to_series()):
                table.index = table.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
            for column in table.columns:
                if pd.api.types.is_datetime64_any_dtype(table[column]):
                    table[column] = table[column].dt.strftime("%Y-%m-%d %H:%M:%S")
            data[ticker] = table.reset_index(drop=False).to_dict(orient="records")
        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
