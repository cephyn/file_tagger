from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

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

def init_db():
    engine = create_engine('sqlite:///file_tags.db')
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()