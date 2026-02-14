import sys
import logging
from sqlmodel import Session, create_engine, select
from googleapiclient.discovery import build
from groq import Groq
import httpx
from google import genai
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("health_check")

def check_youtube():
    try:
        youtube = build('youtube', 'v3', developerKey=settings.youtube_api_key)
        youtube.channels().list(part='id', id='UC_x5XG1OV2P6uYZ5gz9LxKQ').execute()
        logger.info("✅ YouTube API: OK")
        return True
    except Exception as e:
        logger.error(f"❌ YouTube API: FAILED - {e}")
        return False

def check_groq():
    try:
        client = Groq(api_key=settings.groq_api_key)
        client.models.list()
        logger.info("✅ Groq API: OK")
        return True
    except Exception as e:
        logger.error(f"❌ Groq API: FAILED - {e}")
        return False

def check_gemini():
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        client.models.list(config={'page_size': 1})
        logger.info("✅ Gemini API: OK")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini API: FAILED - {e}")
        return False

def check_discord():
    try:
        response = httpx.get(settings.discord_webhook_url)
        if response.status_code in [200, 204, 405]: # 405 is fine since we are doing a GET on a POST endpoint
            logger.info("✅ Discord Webhook: OK")
            return True
        else:
            logger.error(f"❌ Discord Webhook: FAILED - Status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Discord Webhook: FAILED - {e}")
        return False

def check_database():
    try:
        engine = create_engine(f"sqlite:///{settings.db_path}")
        from main import InsightRecord
        with Session(engine) as session:
            session.exec(select(InsightRecord).limit(1)).all()
        logger.info("✅ Database: OK")
        return True
    except Exception as e:
        logger.error(f"❌ Database: FAILED - {e}")
        return False

def main():
    logger.info("Starting health check...")
    results = [
        check_youtube(),
        check_groq(),
        check_gemini(),
        check_discord(),
        check_database()
    ]
    
    if all(results):
        logger.info("🚀 All systems go!")
        sys.exit(0)
    else:
        logger.error("⚠️ Some checks failed. Please check your configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
