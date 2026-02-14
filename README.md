# YouTube to Discord Pipeline

A hardened production-grade pipeline that extracts research-quality insights from YouTube transcripts and drafts creative LinkedIn posts.

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
