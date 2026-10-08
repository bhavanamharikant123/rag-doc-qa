from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "vectorstore"
    collection_name: str = "docs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_path(self) -> Path:
        return BASE_DIR / self.chroma_dir

    @property
    def raw_data_path(self) -> Path:
        return BASE_DIR / "data" / "raw"


settings = Settings()