from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class SoftwareTypeEnum(enum.Enum):
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

class AnalysisStatusEnum(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    projects = relationship("Project", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="api_keys")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="projects")
    software = relationship("Software", back_populates="project")

class Software(Base):
    __tablename__ = "software"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(SQLEnum(SoftwareTypeEnum), nullable=False)
    original_filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    project = relationship("Project", back_populates="software")
    analyses = relationship("Analysis", back_populates="software")
    mimicries = relationship("Mimicry", back_populates="software")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    software_id = Column(String, ForeignKey("software.id"), nullable=False)
    status = Column(SQLEnum(AnalysisStatusEnum), default=AnalysisStatusEnum.PENDING)
    decompiled_paths = Column(JSON, default=list)
    spec_paths = Column(JSON, default=list)
    reconstructed_paths = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    software = relationship("Software", back_populates="analyses")
    functions = relationship("Function", back_populates="analysis")
    classes = relationship("Class", back_populates="analysis")

class Function(Base):
    __tablename__ = "functions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    name = Column(String, nullable=False)
    signature = Column(String, nullable=True)
    return_type = Column(String, nullable=True)
    complexity = Column(Float, nullable=True)
    source_file = Column(String, nullable=True)
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)
    
    analysis = relationship("Analysis", back_populates="functions")

class Class(Base):
    __tablename__ = "classes"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    name = Column(String, nullable=False)
    superclasses = Column(JSON, default=list)
    methods = Column(JSON, default=list)
    properties = Column(JSON, default=list)
    source_file = Column(String, nullable=True)
    
    analysis = relationship("Analysis", back_populates="classes")

class Mimicry(Base):
    __tablename__ = "mimicries"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    software_id = Column(String, ForeignKey("software.id"), nullable=False)
    target_language = Column(String, nullable=False)
    status = Column(SQLEnum(AnalysisStatusEnum), default=AnalysisStatusEnum.PENDING)
    output_paths = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    software = relationship("Software", back_populates="mimicries")

class Sandbox(Base):
    __tablename__ = "sandboxes"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    mimicry_id = Column(String, ForeignKey("mimicries.id"), nullable=True)
    container_id = Column(String, nullable=True)
    container_name = Column(String, nullable=True)
    ports = Column(JSON, default=dict)
    status = Column(String, default="created")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    mimicry = relationship("Mimicry")
