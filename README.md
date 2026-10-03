# YouTube to Discord Pipeline

A LangGraph automation project that extracts insights from YouTube transcripts and prepares draft content for Discord delivery. The project is in development; its repository documents setup, dry-run checks, persistence, and monitoring integrations.

[Portfolio case study](https://hamza-khan-portfolio.hamzakhan102003.chatgpt.site/projects/youtube-discord/) · [LinkedIn](https://www.linkedin.com/in/muhammadhamzakhan/)

## Workflow and scope

The workflow connects transcript extraction, insight selection, structured content preparation, persistence, and Discord integration. The setup below includes a dry-run option for checking the pipeline without delivery side effects. No delivery-volume or latency benchmark is claimed.

## 🛠 Tech Stack
- **Engine**: Python 3.12+, LangGraph
- **LLMs**: Groq (Primary: Llama-3.3-70B), Gemini (Fallback: 2.0-Flash)
- **Storage**: SQLite (SQLModel)
- **Metrics**: Prometheus

## 🚀 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configure**:
   - Copy `.env.example` to `.env` and add your API keys.
   - List YouTube channel IDs in `channel_ids.txt`.

3. **Pre-flight Check**:
   ```bash
   python health_check.py
   ```

4. **Run**:
   ```bash
   python main.py
   # Test without side effects
   python main.py --dry-run
   ```

## 🧪 Development
- **health-check**: `python health_check.py`
- **Tests**: `pytest tests/`
- **Logs**: Structured JSON logs in `script.log`.
- **Database**: `data/insights.db`.
