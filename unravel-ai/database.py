from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any, Union
import datetime
import enum

# Enums
class SoftwareType(str, enum.Enum):
    BINARY = "binary"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript" 
    PYTHON = "python"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    ASSEMBLY = "assembly"
    OTHER = "other"

class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        # Add more password validation if needed
        return v

class UserResponse(UserBase):
    id: str
    is_active: bool
    is_admin: bool
    created_at: datetime.datetime

    class Config:
        orm_mode = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Project schemas
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: str
    user_id: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True

# Software schemas
class SoftwareBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: SoftwareType

class SoftwareCreate(SoftwareBase):
    project_id: str

class SoftwareResponse(SoftwareBase):
    id: str
    project_id: str
    original_filename: str
    file_hash: str
    file_size: int
    created_at: datetime.datetime

    class Config:
        orm_mode = True

# Analysis schemas
class AnalysisBase(BaseModel):
    software_id: str

class AnalysisCreate(AnalysisBase):
    pass

class FunctionResponse(BaseModel):
    id: str
    name: str
    signature: Optional[str]
    return_type: Optional[str]
    complexity: Optional[float]
    source_file: Optional[str]
    
    class Config:
        orm_mode = True

class ClassResponse(BaseModel):
    id: str
    name: str
    superclasses: List[str]
    methods: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    source_file: Optional[str]
    
    class Config:
        orm_mode = True

class AnalysisResponse(AnalysisBase):
    id: str
    status: AnalysisStatus
    decompiled_paths: List[str]
    spec_paths: List[str]
    reconstructed_paths: List[str]
    metadata: Dict[str, Any]
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime]
    error_message: Optional[str]
    functions: List[FunctionResponse] = []
    classes: List[ClassResponse] = []

    class Config:
        orm_mode = True

# Mimicry schemas
class MimicryBase(BaseModel):
    software_id: str
    target_language: str

class MimicryCreate(MimicryBase):
    pass

class MimicryResponse(MimicryBase):
    id: str
    status: AnalysisStatus
    output_paths: List[str]
    metadata: Dict[str, Any]
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime]
    error_message: Optional[str]

    class Config:
        orm_mode = True

# Sandbox schemas
class SandboxBase(BaseModel):
    mimicry_id: Optional[str] = None

class SandboxCreate(SandboxBase):
    pass

class SandboxResponse(SandboxBase):
    id: str
    container_id: Optional[str]
    container_name: Optional[str]
    ports: Dict[str, int]
    status: str
    created_at: datetime.datetime
    expires_at: Optional[datetime.datetime]

    class Config:
        orm_mode = True
