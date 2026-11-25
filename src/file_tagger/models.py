from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey, JSON, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime
from typing import Any
from pathlib import Path

from .infrastructure.app_paths import AppPaths

Base: Any = declarative_base()

# Association table for many-to-many relationship between files and tags
file_tags = Table(
    'file_tags',
    Base.metadata,
    Column('file_id', Integer, ForeignKey('files.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

class File(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    tags = relationship('Tag', secondary=file_tags, back_populates='files')

class Tag(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    color = Column(String, default='#808080')  # Default tag color
    files = relationship('File', secondary=file_tags, back_populates='tags')

class TagSuggestionCache(Base):
    __tablename__ = 'tag_suggestion_cache'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    file_hash = Column(String)  # Store file hash to detect changes
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    suggestions = Column(JSON)  # Store suggestions with confidence scores
    provider = Column(String)  # Store which AI provider made these suggestions

def init_db(database_path: Path | None = None):
    resolved_path = (
        AppPaths.discover().legacy_database
        if database_path is None
        else database_path
    ).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{resolved_path.as_posix()}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()