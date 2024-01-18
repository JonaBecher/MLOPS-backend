from typing import Dict

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, TIMESTAMP, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, joinedload

engine = create_engine('sqlite:///db.db', pool_size=20, max_overflow=0)
Session = sessionmaker(bind=engine)

Base = declarative_base()


class Model(Base):
    __tablename__ = 'models'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    active = Column(Boolean, nullable=True)
    projectId = Column(Integer, ForeignKey('projects.id'))
    metrics = relationship("ModelMetrics", backref="model", lazy='select')


class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    data = Column(String)
    image = Column(String)
    modelId = Column(Integer, ForeignKey('models.id'))
    deviceId = Column(Integer, ForeignKey('devices.id'))


class Project(Base):
    __tablename__ = 'projects'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(Integer)
    current_modelId = Column(String, ForeignKey('models.id'))
    current_model = relationship("Model", foreign_keys=[current_modelId],
                                 primaryjoin="Model.id == Project.current_modelId")


class Device(Base):
    __tablename__ = 'devices'
    id = Column(Integer, primary_key=True, autoincrement=True)
    projectId = Column(Integer, ForeignKey('projects.id'))
    modelId = Column(Integer, ForeignKey('models.id'))
    last_online = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    project = relationship("Project")
    model = relationship("Model")
    project = relationship("Project", foreign_keys=[projectId],
                                 primaryjoin="Device.projectId == Project.id")


def setup_db():
    Base.metadata.create_all(engine)


def get_entity(table: Base, id: int | str | None, filter_attr="id", first=True, option=None):
    session = Session()
    tmp = session.query(table)
    if option is not None:
        tmp = tmp.options(joinedload(getattr(table, option)))
    if id is not None:
        tmp = tmp.filter(getattr(table, filter_attr) == id)
        if first:
            entity = tmp.first()
            print("first")
        else:
            print("all")
            entity = tmp.all()
    else:
        print("else")
        entity = tmp.all()
    session.close()
    return entity


def insert_or_update_entity(table: Base, id: int | str | None, values: Dict):
    session = Session()
    entity = None
    if id is not None:
        entity = session.query(table).filter(table.id == id).first()
    if id is None or not entity:
        print(f"insert entity. id: {id}. tabel: {table}")
        new_entity = table()
        if id is not None:
            new_entity.id = id
        for key, value in values.items():
            if hasattr(new_entity, key):
                setattr(new_entity, key, value)
        session.add(new_entity)
        session.commit()
        return new_entity.id
    else:
        update_entity(table, id, values, entity_to_update=entity, session=session)


def update_entity(table: Base, id: int, values_to_update: Dict, entity_to_update=None, session=None):
    print(f"update entity. id: {id}. tabel: {table}. values: {values_to_update}. entity: {entity_to_update}")
    if not session:
        session = Session()
    if entity_to_update is None:
        entity_to_update = session.query(table).filter(table.id == id).first()
    if entity_to_update:
        for key, value in values_to_update.items():
            if hasattr(entity_to_update, key):
                setattr(entity_to_update, key, value)
        session.commit()
        return id

    session.close()
    return None
