from typing import List

from app import db
from app.auth.jwt import get_current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from . import schema, services, validator

router = APIRouter(tags=["Users"], prefix="/user")


@router.post("/create_user", status_code=status.HTTP_201_CREATED)
async def create_user_registration(
    request: schema.User, database: Session = Depends(db.get_db)
):
    """ Register a new user
    
        Verify if the email exist, and if already exist throw a Exception else use the service to create the new user
    
    """
    
    if(await validator.verify_email_exist(request.email, database)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"User already exist!")
    
    new_user = await services.new_user_register(request, database)

    return new_user


@router.get("/")
async def get_all_users(
    database: Session = Depends(db.get_db),
    current_user: schema.User = Depends(get_current_user),
):
    return await services.all_users(database)


@router.get("/{id}", response_model=schema.DisplayUser)
async def get_user_by_id(
    id: int,
    database: Session = Depends(db.get_db),
    current_user: schema.User = Depends(get_current_user),
):
    return await services.get_user_by_id(id, database)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_by_id(
    id: int,
    database: Session = Depends(db.get_db),
    current_user: schema.User = Depends(get_current_user),
):
    return await services.delete_user_by_id(id, database)
