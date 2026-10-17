from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# Ensure storage directory exists
db_path = settings.BASE_DIR / "storage"
db_path.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.LOG_LEVEL.lower() == "debug"),
    future=True,
)

async_session = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
