# YouTube to Discord Transcript Processor

A robust Python-based tool that monitors YouTube channels, extracts transcripts, generates AI-powered summaries, and sends notifications to Discord.

## 🛠 Tech & Tools

- **Core**: Python 3.12+
- **AI Models**: 
  - **Groq** (Primary): Llama 3.3 70B for fast, high-quality summaries.
  - **Google Gemini** (Fallback): Gemini 2.5 Flash for reliable redundancy.
- **Data Processing**: **Polars** for efficient CSV-based data management.
- **APIs**:
  - **YouTube Data API v3**: For fetching latest channel videos.
  - **YouTube Transcript API**: For extracting video subtitles.
  - **Discord Webhooks**: For automated notifications.
- **Utilities**: `python-dotenv`, `google-api-python-client`, `google-genai`.

## 🚀 Setup Instructions

### 1. Clone & Install
```bash
git clone https://github.com/Muhammad-Hamza-Khan-03/youtube_to_discord.git
cd youtube_to_discord

# setup the environment
chmod +x setup.sh
./setup.sh
```

### 2. Add Channel IDs and Env variables
Add the YouTube Channel IDs you want to monitor to `channel_ids.txt` (one per line) and the environment variables to `.env` file.

### 3. Run the Script
```bash
source .venv/bin/activate
python main.py
```

## 📝 How it Works
1. **Fetch**: Checks `channel_ids.txt` for new videos uploaded in the last 24 hours.
2. **Filter**: Skips videos already present in `data/data.csv`.
3. **Extract**: Retrieves the transcript using `youtube-transcript-api`.
4. **Summarize**: Attempts to generate a summary using Groq; falls back to Gemini if Groq fails.
5. **Notify**: Sends the formatted summary and metadata to your Discord channel.
