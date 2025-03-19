"""
Kaleidoscope AI - Main FastAPI Application
"""

import os
import uuid
import logging
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, BackgroundTasks, Form, Query, Path, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import asyncio
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr, validator

from src.db.database import get_db, Base, engine
from src.db.models import User, Software, DecompiledFile, SpecFile, ReconstructedSoftware, MimickedSoftware, SandboxInstance, Subscription
from src.core.ingestion import SoftwareIngestionManager
from src.core.sandbox import SandboxManager
from src.utils.config import config, get
from src.core.detection import FileType

# Initialize logging
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Kaleidoscope AI",
    description="Software ingestion, analysis, and mimicry system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Static files
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserResponse(UserBase):
    user_id: str
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        orm_mode = True

class SoftwareBase(BaseModel):
    name: str
    description: Optional[str] = None

class SoftwareCreate(SoftwareBase):
    pass

class SoftwareResponse(SoftwareBase):
    software_id: str
    user_id: str
    original_filename: str
    file_type: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SandboxConfig(BaseModel):
    name: str
    type: str = "web"
    language: str
    framework: Optional[str] = ""
    memory_limit: Optional[str] = "2g"
    cpu_limit: Optional[float] = 1.0
    network_enabled: Optional[bool] = True
    expose_ports: Optional[List[int]] = []
    environment: Optional[Dict[str, str]] = {}
    timeout_seconds: Optional[int] = 7200

class MimicryConfig(BaseModel):
    target_language: str

# User authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.user_id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = db.query(User).filter(User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        user_id=str(uuid.uuid4()),
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
        is_active=True,
        is_admin=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/software", response_model=SoftwareResponse)
async def upload_software(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if file extension is allowed
    allowed_extensions = get("ALLOWED_EXTENSIONS", [])
    file_ext = os.path.splitext(file.filename)[1].lower().strip(".")
    if file_ext not in allowed_extensions and "*" not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File extension not allowed. Allowed extensions: {', '.join(allowed_extensions)}")
    
    # Generate file path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("uploads", unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create software record
    software = Software(
        software_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        name=name,
        description=description,
        original_filename=file.filename,
        storage_path=file_path,
        status="pending"
    )
    db.add(software)
    db.commit()
    db.refresh(software)
    
    # Start background processing
    ingestion_manager = SoftwareIngestionManager(db)
    background_tasks.add_task(
        ingestion_manager.process_software,
        software.software_id,
        file_path
    )
    
    return software

@app.get("/software", response_model=List[SoftwareResponse])
async def list_software(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    software = db.query(Software)\
        .filter(Software.user_id == current_user.user_id)\
        .offset(skip)\
        .limit(limit)\
        .all()
    return software

@app.get("/software/{software_id}", response_model=SoftwareResponse)
async def get_software(
    software_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    return software

@app.get("/software/{software_id}/decompiled")
async def list_decompiled_files(
    software_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if software exists and belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Get decompiled files
    files = db.query(DecompiledFile)\
        .filter(DecompiledFile.software_id == software_id)\
        .all()
    
    return [
        {
            "file_id": file.file_id,
            "file_path": os.path.basename(file.file_path),
            "file_type": file.file_type,
            "created_at": file.created_at
        }
        for file in files
    ]

@app.get("/software/{software_id}/specs")
async def list_spec_files(
    software_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if software exists and belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Get spec files
    files = db.query(SpecFile)\
        .filter(SpecFile.software_id == software_id)\
        .all()
    
    return [
        {
            "spec_id": file.spec_id,
            "file_path": os.path.basename(file.file_path),
            "created_at": file.created_at
        }
        for file in files
    ]

@app.get("/software/{software_id}/reconstructed")
async def get_reconstructed_software(
    software_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if software exists and belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Get reconstructed software
    reconstructed = db.query(ReconstructedSoftware)\
        .filter(ReconstructedSoftware.software_id == software_id)\
        .first()
    
    if not reconstructed:
        raise HTTPException(status_code=404, detail="Reconstructed software not found")
    
    # Get list of files in directory
    directory = reconstructed.directory_path
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory)
            file_list.append({
                "path": rel_path,
                "type": "file"
            })
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            rel_path = os.path.relpath(dir_path, directory)
            file_list.append({
                "path": rel_path,
                "type": "directory"
            })
    
    return {
        "reconstructed_id": reconstructed.reconstructed_id,
        "directory_path": reconstructed.directory_path,
        "created_at": reconstructed.created_at,
        "status": reconstructed.status,
        "files": file_list
    }

@app.post("/software/{software_id}/mimic", response_model=dict)
async def create_mimicked_software(
    software_id: str,
    config: MimicryConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if software exists and belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Check if software processing is complete
    if software.status != "completed":
        raise HTTPException(status_code=400, detail="Software processing not completed")
    
    # Get spec files
    spec_files = db.query(SpecFile)\
        .filter(SpecFile.software_id == software_id)\
        .all()
    
    if not spec_files:
        raise HTTPException(status_code=400, detail="No specification files available")
    
    # Create mimicked software record
    mimicked_id = str(uuid.uuid4())
    mimicked = MimickedSoftware(
        mimicked_id=mimicked_id,
        software_id=software_id,
        target_language=config.target_language,
        directory_path="",  # Will be updated during processing
        status="pending"
    )
    db.add(mimicked)
    db.commit()
    
    # Start background processing
    ingestion_manager = SoftwareIngestionManager(db)
    background_tasks.add_task(
        ingestion_manager.mimic_software,
        software_id,
        mimicked_id,
        [f.file_path for f in spec_files],
        config.target_language
    )
    
    return {
        "mimicked_id": mimicked_id,
        "software_id": software_id,
        "target_language": config.target_language,
        "status": "pending"
    }

@app.get("/software/{software_id}/mimicked")
async def list_mimicked_software(
    software_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if software exists and belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Get mimicked software
    mimicked = db.query(MimickedSoftware)\
        .filter(MimickedSoftware.software_id == software_id)\
        .all()
    
    return [
        {
            "mimicked_id": m.mimicked_id,
            "target_language": m.target_language,
            "created_at": m.created_at,
            "status": m.status
        }
        for m in mimicked
    ]

@app.get("/mimicked/{mimicked_id}")
async def get_mimicked_software(
    mimicked_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Get mimicked software
    mimicked = db.query(MimickedSoftware).filter(MimickedSoftware.mimicked_id == mimicked_id).first()
    if not mimicked:
        raise HTTPException(status_code=404, detail="Mimicked software not found")
    
    # Check if software belongs to user
    software = db.query(Software)\
        .filter(Software.software_id == mimicked.software_id, Software.user_id == current_user.user_id)\
        .first()
    if not software:
        raise HTTPException(status_code=404, detail="Software not found")
    
    # Get list of files in directory
    directory = mimicked.directory_path
    if not os.path.exists(directory):
        return {
            "mimicked_id": mimicked.mimicked_id,
            "software_id": mimicked.software_id,
            "target_language": mimicked.target_language,
            "status": mimicked.status,
            "created_at": mimicked.created_at,
            "files": []
        }
    
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory)
            file_list.append({
                "path": rel_path,
                "type": "file"
            })
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            rel_path = os.path.relpath(dir_path, directory)
            file_list.append({
                "path": rel_path,
                "type": "directory"
            })
    
    return {
        "mimicked_id": mimicked.mimicked_id,
        "software_id": mimicked.software_id,
        "target_language": mimicked.target_language,
        "status": mimicked.status,
        "created_at": mimicked.created_at,
        "files": file_list
    }

@app.post("/sandbox/create")
async def create_sandbox(
    config: SandboxConfig,
    software_id: Optional[str] = None,
    mimicked_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if not software_id and not mimicked_id:
        raise HTTPException(status_code=400, detail="Either software_id or mimicked_id must be provided")
    
    # Get directory path
    directory_path = None
    if software_id:
        # Check if software exists and belongs to user
        software = db.query(Software)\
            .filter(Software.software_id == software_id, Software.user_id == current_user.user_id)\
            .first()
        if not software:
            raise HTTPException(status_code=404, detail="Software not found")
        
        # Get reconstructed software
        reconstructed = db.query(ReconstructedSoftware)\
            .filter(ReconstructedSoftware.software_id == software_id)\
            .first()
        if not reconstructed:
            raise HTTPException(status_code=404, detail="Reconstructed software not found")
        
        directory_path = reconstructed.directory_path
    elif mimicked_id:
        # Get mimicked software
        mimicked = db.query(MimickedSoftware).filter(MimickedSoftware.mimicked_id == mimicked_id).first()
        if not mimicked:
            raise HTTPException(status_code=404, detail="Mimicked software not found")
        
        # Check if software belongs to user
        software = db.query(Software)\
            .filter(Software.software_id == mimicked.software_id, Software.user_id == current_user.user_id)\
            .first()
        if not software:
            raise HTTPException(status_code=404, detail="Software not found")
        
        directory_path = mimicked.directory_path
    
    if not directory_path or not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="Software directory not found")
    
    # Create sandbox
    sandbox_manager = SandboxManager(db)
    sandbox_info = sandbox_manager.create_sandbox(
        user_id=current_user.user_id,
        app_dir=directory_path,
        app_config=config.dict(),
        software_id=software_id,
        mimicked_id=mimicked_id
    )
    
    return sandbox_info

@app.get("/sandbox/{sandbox_id}")
async def get_sandbox_status(
    sandbox_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if sandbox exists and belongs to user
    sandbox = db.query(SandboxInstance)\
        .filter(SandboxInstance.sandbox_id == sandbox_id, SandboxInstance.user_id == current_user.user_id)\
        .first()
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    # Get sandbox status
    sandbox_manager = SandboxManager(db)
    status = sandbox_manager.get_sandbox_status(sandbox_id)
    
    return status

@app.get("/sandbox/{sandbox_id}/logs")
async def get_sandbox_logs(
    sandbox_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if sandbox exists and belongs to user
    sandbox = db.query(SandboxInstance)\
        .filter(SandboxInstance.sandbox_id == sandbox_id, SandboxInstance.user_id == current_user.user_id)\
        .first()
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    # Get sandbox logs
    sandbox_manager = SandboxManager(db)
    logs = sandbox_manager.get_container_logs(sandbox_id)
    
    return {"logs": logs}

@app.post("/sandbox/{sandbox_id}/stop")
async def stop_sandbox(
    sandbox_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if sandbox exists and belongs to user
    sandbox = db.query(SandboxInstance)\
        .filter(SandboxInstance.sandbox_id == sandbox_id, SandboxInstance.user_id == current_user.user_id)\
        .first()
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    # Stop sandbox
    sandbox_manager = SandboxManager(db)
    success = sandbox_manager.stop_sandbox(sandbox_id)
    
    return {"success": success}

@app.get("/files/{file_type}/{file_id}")
async def get_file_content(
    file_type: str,
    file_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Get file based on type
    file_path = None
    if file_type == "decompiled":
        file = db.query(DecompiledFile).filter(DecompiledFile.file_id == file_id).first()
        if file:
            # Check if user has access to this file
            software = db.query(Software)\
                .filter(Software.software_id == file.software_id, Software.user_id == current_user.user_id)\
                .first()
            if software:
                file_path = file.file_path
    elif file_type == "spec":
        file = db.query(SpecFile).filter(SpecFile.spec_id == file_id).first()
        if file:
            # Check if user has access to this file
            software = db.query(Software)\
                .filter(Software.software_id == file.software_id, Software.user_id == current_user.user_id)\
                .first()
            if software:
                file_path = file.file_path
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return file content
    return FileResponse(file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
