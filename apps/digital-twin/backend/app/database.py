from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "digital_twin.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


def ensure_session_vision_gru_columns() -> None:
    """Add GRU result columns to an existing SQLite database in place."""
    inspector = inspect(engine)
    if not inspector.has_table("session_vision_samples"):
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("session_vision_samples")
    }
    statements = []
    if "gru_activity" not in columns:
        statements.append(
            "ALTER TABLE session_vision_samples "
            "ADD COLUMN gru_activity VARCHAR NOT NULL DEFAULT 'unknown'"
        )
    if "gru_confidence" not in columns:
        statements.append(
            "ALTER TABLE session_vision_samples "
            "ADD COLUMN gru_confidence FLOAT NOT NULL DEFAULT 0.0"
        )
    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

