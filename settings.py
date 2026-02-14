from pathlib import Path
from typing import Set, Set
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    youtube_api_key: str = Field(..., alias="YOUTUBE_API_KEY")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    discord_webhook_url: str = Field(..., alias="DISCORD_WEBHOOK_URL")
    
    # Model Config
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    
    # Proxy Config (Optional)
    webshare_username: str = Field(None, alias="WEBSHARE_USERNAME")
    webshare_password: str = Field(None, alias="WEBSHARE_PASSWORD")
    development_mode: bool = Field(False, alias="DEVELOPMENT_MODE")
    
    # Paths
    data_dir: Path = Path("data")
    db_path: Path = Path("data/insights.db")
    channel_ids_file: Path = Path("channel_ids.txt")
    log_file: Path = Path("script.log")
    
    # Workflow Magic Numbers
    score_threshold: float = 1.8
    transcript_truncation_limit: int = 4000
    max_youtube_results: int = 50
    discord_chunk_size: int = 1900
    lookback_hours: int = 24
    
    # Concurrency
    max_workers: int = 5
    
    # LLM Settings
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    llm_retry_attempts: int = 3
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')

settings = Settings()
