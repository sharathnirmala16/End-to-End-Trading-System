import jwt

from jwt.exceptions import InvalidTokenError
from utils import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM
from credentials import JWT_SECRET_KEY
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Any, Optional


class JWTBearer(HTTPBearer):
    @staticmethod
    def decodeJWT(jwt_token: str) -> Any | None:
        try:
            payload = jwt.decode(jwt_token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except:
            return None

    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:
        credentials: HTTPAuthorizationCredentials = await super(
            JWTBearer, self
        ).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=403, detail="Invalid authentication scheme."
                )
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=403, detail="Invalid token or expired token."
                )
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwt_token: str) -> bool:
        is_token_valid: bool = False

        try:
            payload = JWTBearer.decodeJWT(jwt_token)
        except:
            payload = None

        if payload:
            is_token_valid = True
        return is_token_valid


jwt_bearer = JWTBearer()
