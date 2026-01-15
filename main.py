import os
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

import polars as pl
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from googleapiclient.discovery import build
from groq import Groq
from discordwebhook import Discord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('script.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class VideoData:
    """Data class for video information"""
    channel_name: str
    video_title: str
    video_id: str
    transcript: str
    summary: str
    processed_date: str


class Config:
    """Configuration management"""
    def __init__(self):
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        self.data_dir = Path('data')
        self.data_csv = self.data_dir / 'data.csv'
        self.channel_ids_file = Path('channel_ids.txt')
        
        self.validate()
    
    def validate(self):
        """Validate configuration"""
        if not self.youtube_api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable not set")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        if not self.discord_webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL environment variable not set")
        if not self.channel_ids_file.exists():
            self.channel_ids_file.parent.mkdir(parents=True, exist_ok=True)
            # raise FileNotFoundError(f"Channel IDs file not found: {self.channel_ids_file}")


class DataManager:
    """Manage CSV data operations using Polars LazyAPI"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Create CSV with headers if it doesn't exist"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists():
            df = pl.DataFrame({
                'channel_name': [],
                'video_title': [],
                'video_id': [],
                'transcript': [],
                'summary': [],
                'processed_date': []
            })
            df.write_csv(self.csv_path)
            logger.info(f"Created new CSV file: {self.csv_path}")
    
    def get_existing_video_ids(self) -> Set[str]:
        """Get set of existing video IDs from CSV"""
        try:
            df = pl.scan_csv(self.csv_path)
            video_ids = df.select('video_id').collect()['video_id'].to_list()
            return set(video_ids)
        except Exception as e:
            logger.error(f"Error reading existing video IDs: {e}")
            return set()
    
    def append_video_data(self, video_data: VideoData):
        """Append new video data to CSV"""
        try:
            new_row = pl.DataFrame({
                'channel_name': [video_data.channel_name],
                'video_title': [video_data.video_title],
                'video_id': [video_data.video_id],
                'transcript': [video_data.transcript],
                'summary': [video_data.summary],
                'processed_date': [video_data.processed_date]
            })
            
            # Read existing data and concatenate
            if self.csv_path.stat().st_size > 0:
                existing_df = pl.read_csv(self.csv_path)
                updated_df = pl.concat([existing_df, new_row])
            else:
                updated_df = new_row
            
            updated_df.write_csv(self.csv_path)
            logger.info(f"Appended video {video_data.video_id} to CSV")
            
        except Exception as e:
            logger.error(f"Error appending video data: {e}")
            raise


class YouTubeService:
    """Handle YouTube API operations"""
    
    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.transcript_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=os.getenv('WEBSHARE_USERNAME'),
                proxy_password=os.getenv('WEBSHARE_PASSWORD'),
            )
        )
    
    def _retry_api_call(self, func, *args, **kwargs):
        """Helper to retry API calls on failure"""
        max_retries = 3
        delay = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs).execute()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"API call failed, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)

    def get_channel_name(self, channel_id: str) -> str:
        """Fetch channel name from channel ID"""
        try:
            response = self._retry_api_call(
                self.youtube.channels().list,
                part='snippet',
                id=channel_id
            )
            
            if response['items']:
                return response['items'][0]['snippet']['title']
            return "Unknown Channel"
        except Exception as e:
            logger.error(f"Error fetching channel name for {channel_id}: {e}")
            return "Unknown Channel"
    
    def get_latest_videos(self, channel_id: str, hours: int = 24) -> List[Dict[str, str]]:
        """Fetch videos uploaded in the last N hours from a channel"""
        try:
            # Calculate time threshold
            published_after = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + 'Z'
            
            response = self._retry_api_call(
                self.youtube.search().list,
                part='snippet',
                channelId=channel_id,
                publishedAfter=published_after,
                order='date',
                type='video',
                maxResults=50
            )
            
            videos = []
            for item in response.get('items', []):
                videos.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_name': item['snippet']['channelTitle']
                })
            
            logger.info(f"Found {len(videos)} videos from channel {channel_id}")
            return videos
            
        except Exception as e:
            logger.error(f"Error fetching videos from channel {channel_id}: {e}")
            return []
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Extract transcript from video"""
        max_retries = 3
        delay = 3
        for attempt in range(max_retries):
            try:
                transcript_obj = self.transcript_api.fetch(video_id)
                
                # Combine all snippets into a single transcript
                transcript_text = ' '.join([
                    snippet.text for snippet in transcript_obj.snippets
                ])
                
                logger.info(f"Successfully fetched transcript for video {video_id}")
                return transcript_text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching transcript for video {video_id} after {max_retries} attempts: {e}")
                    return None
                logger.warning(f"Transcript fetch failed, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)


class SummaryGenerator:
    """Generate summaries using Groq API"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def generate_summary(self, transcript: str, video_title: str) -> Optional[str]:
        """Generate detailed summary with keywords from transcript"""
        try:
            prompt = f"""
You are a professional content summarizer. Given the following video transcript, create a detailed summary that:

1. Highlights all important points and key takeaways
2. Identifies and emphasizes important keywords and concepts (use **bold** for keywords)
3. Structures the content in a way that's engaging and professional
4. Makes it suitable for creating interesting social media posts

Video Title: {video_title}

Transcript:
{transcript}

Please provide a comprehensive summary that captures the essence of the content while making it engaging and informative.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert content summarizer who creates engaging, professional summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            summary = response.choices[0].message.content
            logger.info(f"Successfully generated summary for video: {video_title}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None


class DiscordNotifier:
    """Send notifications to Discord"""
    
    def __init__(self, webhook_url: str):
        self.discord = Discord(url=webhook_url)
    
    def send_summary(self, video_data: VideoData) -> bool:
        """Send video summary to Discord"""
        try:
            processed_dt = datetime.fromisoformat(video_data.processed_date)
            date_str = processed_dt.strftime("%A, %Y-%m-%d")
            
            message = f"""
🎥 **New Video Processed!**

**Channel:** {video_data.channel_name}
**Title:** {video_data.video_title}
**Date:** {date_str}
**Video ID:** {video_data.video_id}
**Video URL:** https://www.youtube.com/watch?v={video_data.video_id}

**Summary:**
{video_data.summary}

---
*Processed on {video_data.processed_date}*
"""
            
            # Discord has a 2000 character limit, split if necessary
            if len(message) > 2000:
                # Send in chunks
                chunks = [message[i:i+1900] for i in range(0, len(message), 1900)]
                for chunk in chunks:
                    self.discord.post(content=chunk)
            else:
                self.discord.post(content=message)
            
            logger.info(f"Successfully sent Discord notification for video {video_data.video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False


class TranscriptProcessor:
    """Main processor orchestrating all operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config.data_csv)
        self.youtube_service = YouTubeService(config.youtube_api_key)
        self.summary_generator = SummaryGenerator(config.groq_api_key)
        self.discord_notifier = DiscordNotifier(config.discord_webhook_url)
    
    def load_channel_ids(self) -> List[str]:
        """Load channel IDs from file"""
        try:
            with open(self.config.channel_ids_file, 'r') as f:
                channel_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(channel_ids)} channel IDs")
            return channel_ids
        except Exception as e:
            logger.error(f"Error loading channel IDs: {e}")
            return []
    
    def process_video(self, video_info: Dict[str, str]) -> bool:
        """Process a single video: extract transcript, generate summary, store data"""
        video_id = video_info['video_id']
        
        # Get transcript
        transcript = self.youtube_service.get_transcript(video_id)
        if not transcript:
            logger.warning(f"Skipping video {video_id} - no transcript available")
            return False
        
        # Generate summary
        summary = self.summary_generator.generate_summary(
            transcript, 
            video_info['title']
        )
        if not summary:
            logger.warning(f"Skipping video {video_id} - summary generation failed")
            return False
        
        # Create video data object
        video_data = VideoData(
            channel_name=video_info['channel_name'],
            video_title=video_info['title'],
            video_id=video_id,
            transcript=transcript,
            summary=summary,
            processed_date=datetime.now().isoformat()
        )
        
        # Store in CSV
        try:
            self.data_manager.append_video_data(video_data)
        except Exception as e:
            logger.error(f"Failed to store video data for {video_id}: {e}")
            return False
        
        # Send to Discord
        self.discord_notifier.send_summary(video_data)
        
        return True
    
    def run(self):
        """Main execution flow"""
        logger.info("=" * 50)
        logger.info("Starting YouTube Transcript Processing")
        logger.info("=" * 50)
        
        # Load channel IDs
        channel_ids = self.load_channel_ids()
        if not channel_ids:
            logger.error("No channel IDs found. Exiting.")
            return
        
        # Get existing video IDs
        existing_video_ids = self.data_manager.get_existing_video_ids()
        logger.info(f"Found {len(existing_video_ids)} existing videos in database")
        
        # Process each channel
        total_processed = 0
        total_new_videos = 0
        
        for channel_id in channel_ids:
            logger.info(f"Processing channel: {channel_id}")
            
            # Get latest videos
            latest_videos = self.youtube_service.get_latest_videos(channel_id)
            
            # Filter out existing videos
            new_videos = [
                v for v in latest_videos 
                if v['video_id'] not in existing_video_ids
            ]
            
            total_new_videos += len(new_videos)
            logger.info(f"Found {len(new_videos)} new videos to process")
            
            # Process each new video
            for video_info in new_videos:
                logger.info(f"Processing video: {video_info['title']} ({video_info['video_id']})")
                
                if self.process_video(video_info):
                    total_processed += 1
                    logger.info(f"Successfully processed video {video_info['video_id']}")
                else:
                    logger.warning(f"Failed to process video {video_info['video_id']}")
            time.sleep(60)
        logger.info("=" * 50)
        logger.info(f"Processing complete!")
        logger.info(f"New videos found: {total_new_videos}")
        logger.info(f"Successfully processed: {total_processed}")
        logger.info("=" * 50)


def main():
    """Entry point"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create and run processor
        processor = TranscriptProcessor(config)
        processor.run()
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()