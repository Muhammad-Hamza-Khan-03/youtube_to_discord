import sys
import json
import string
import html
import logging
import asyncio
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import List, Dict, Optional, Set, Literal, TypedDict, Any
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
import spacy
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from googleapiclient.discovery import build
from groq import Groq
from discordwebhook import Discord
from google import genai
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from tenacity import retry, wait_exponential, stop_after_attempt
from pythonjsonlogger import jsonlogger
from sqlmodel import SQLModel, Session, create_engine, select, Field as SqlField

from settings import settings
from prompts import (
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    INSIGHT_SCORING_SYSTEM_PROMPT,
    LINKEDIN_DRAFT_SYSTEM_PROMPT
)
from metrics import NODE_EXECUTION_COUNT, NODE_DURATION, REJECTION_COUNT

# -----------------------------
# Versioning
# -----------------------------
__version__ = "1.0.0"

# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging():
    log_file = settings.log_file
    max_bytes = 10_000_000  # 10MB
    backup_count = 5

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console Handler with human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler with JSON format for structured logging
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    log_format = '%(asctime)s %(name)s %(levelname)s %(message)s %(video_id)s %(channel_id)s'
    json_formatter = jsonlogger.JsonFormatter(log_format)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

class VideoContextAdapter(logging.LoggerAdapter):
    """Adapter to inject video_id and channel_id into logs"""
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update({
            "video_id": self.extra.get("video_id", "N/A"),
            "channel_id": self.extra.get("channel_id", "N/A")
        })
        kwargs["extra"] = extra
        return msg, kwargs

# -----------------------------
# Exceptions
# -----------------------------
class LLMParseError(Exception):
    """Custom exception when LLM fails to return valid JSON or schema"""
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response

# -----------------------------
# Models & Schemas
# -----------------------------

# SQLite Models
class InsightRecord(SQLModel, table=True):
    video_id: str = SqlField(primary_key=True)
    channel_name: str
    channel_id: str
    video_title: str
    transcript: str
    compressed_transcript: str
    content_bucket: str
    intended_signal: str
    post_type: Optional[str] = None
    approved_for_post: bool
    post_content: Optional[str] = None
    confidence_score: Optional[float] = None
    academic_depth: Optional[float] = None
    non_obvious: Optional[float] = None
    transferability: Optional[float] = None
    reject_reason: Optional[str] = None
    rejection_stage: Optional[str] = None
    processed_at: datetime = SqlField(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))

class ChannelAudit(SQLModel, table=True):
    channel_id: str = SqlField(primary_key=True)
    last_processed_at: datetime
    consecutive_errors: int = 0
    is_circuit_broken: bool = False

# LLM Schemas
class InsightExtraction(BaseModel):
    """Structured insights extracted from video content"""
    content_bucket: Literal[
        "core_ml_foundations",
        "applied_ml_systems",
        "research_curiosity",
        "learning_correction",
        "REJECT"
    ]
    non_obvious_insight: Optional[str] = None
    corrected_misconception: Optional[str] = None
    mental_model: Optional[str] = None
    open_research_question: Optional[str] = None
    intended_signal: Literal["professor", "recruiter", "both", "none"]
    extraction_notes: Optional[str] = None

class InsightScores(BaseModel):
    """Quality scores for insight validation"""
    academic_depth_score: float = Field(ge=0, le=1)
    non_obvious_score: float = Field(ge=0, le=1)
    transferability_score: float = Field(ge=0, le=1)
    reject_reason: Optional[str] = None

class DraftPost(BaseModel):
    """Final LinkedIn post draft"""
    post_content: str
    confidence_score: float = Field(default=0.0,ge=0, le=1)

class WorkflowState(TypedDict):
    """Complete state tracked through the pipeline"""
    video_id: str
    channel_id: str
    channel_name: str
    video_title: str
    transcript: str
    compressed_transcript: Optional[str]
    insights: Optional[InsightExtraction]
    scores: Optional[InsightScores]
    post_type: Optional[str]
    draft: Optional[DraftPost]
    approved_for_post: bool
    reject_reason: Optional[str]
    rejection_stage: Optional[str]
    processed_at: str

# -----------------------------
# Global Singleton Cache
# -----------------------------
_NLP_CACHE = None

def get_nlp():
    global _NLP_CACHE
    if _NLP_CACHE is None:
        try:
            _NLP_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            logger.critical("Spacy model 'en_core_web_sm' not found. Please install it.", exc_info=True)
            raise
    return _NLP_CACHE

# -----------------------------
# Linguistic Constants
# -----------------------------
NEGATION_WORDS: Set[str] = {"no", "not", "never", "none", "n't"}
DISCOURSE_WORDS: Set[str] = {
    "because", "therefore", "however", "although", "though",
    "but", "instead", "rather", "whereas", "while",
    "if", "when", "unless", "since",
    "assume", "assumption", "suggests", "implies",
    "likely", "unlikely", "possibly", "probably"
}
IMPORTANT_DEP_LABELS: Set[str] = {"ROOT", "nsubj", "dobj", "pobj", "ccomp", "xcomp", "advcl"}
DEFAULT_POS_TO_KEEP: Set[str] = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "NUM", "ADV"}
POST_PROCESS_STOPWORDS: Set[str] = {"is", "am", "are", "s", "es", "i", "this", "but", "he", "she", "it"}

@dataclass(frozen=True)
class CompressionConfig:
    keep_entities: bool = True
    min_tokens_per_sentence: int = 4
    max_sentence_length: int = 40
    pos_to_keep: Set[str] = field(default_factory=lambda: DEFAULT_POS_TO_KEEP)
    remove_punctuation: bool = True
    post_stopwords: Set[str] = field(default_factory=lambda: POST_PROCESS_STOPWORDS)

# -----------------------------
# Helper Services
# -----------------------------

class SemanticTextCompressor:
    """End-to-end semantic text compressor for LLM ingestion."""
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.nlp = get_nlp()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def compress(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        
        doc = self.nlp(text)
        compressed_sentences = []
        for sent in doc.sents:
            sent_tokens = []
            for token in sent:
                if self.config.keep_entities and token.ent_type_:
                    sent_tokens.append(token.text)
                    continue
                if token.lower_ in NEGATION_WORDS:
                    sent_tokens.append(token.text)
                    continue
                if token.lower_ in DISCOURSE_WORDS:
                    sent_tokens.append(token.text)
                    continue
                if token.dep_ in IMPORTANT_DEP_LABELS:
                    sent_tokens.append(token.text)
                    continue
                if token.pos_ in self.config.pos_to_keep and not token.is_stop:
                    sent_tokens.append(token.text)
                    continue
                if token.is_punct or token.is_space:
                    continue
            if len(sent_tokens) >= self.config.min_tokens_per_sentence:
                sent_tokens = sent_tokens[: self.config.max_sentence_length]
                compressed_sentences.append(" ".join(sent_tokens))
        
        compressed_text = " ".join(compressed_sentences)
        processed_text = self._post_process(compressed_text)

        # Token-aware truncation
        if self.tokenizer:
            tokens = self.tokenizer.encode(processed_text)
            if len(tokens) > settings.transcript_truncation_limit:
                processed_text = self.tokenizer.decode(tokens[:settings.transcript_truncation_limit])
        else:
            # Fallback to character truncation if tiktoken fails
            processed_text = processed_text[:settings.transcript_truncation_limit]

        return processed_text

    def _post_process(self, text: str) -> str:
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        words = [word for word in text.split() if word.lower() not in self.config.post_stopwords]
        return " ".join(words)

class LLMService:
    """Wrapper for LLM calls with fallback logic and retries"""
    def __init__(self, groq_api_key: str, gemini_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.groq_model = settings.groq_model
        self.gemini_model = settings.gemini_model

    def call_llm(self, system_prompt: str, user_message: str, response_model: BaseModel) -> BaseModel:
        try:
            return self._call_groq(system_prompt, user_message, response_model)
        except Exception as e:
            logger.warning(f"Groq call failed, falling back to Gemini", exc_info=True)
            return self._call_gemini(system_prompt, user_message, response_model)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(settings.llm_retry_attempts))
    def _call_groq(self, system_prompt: str, user_message: str, response_model: BaseModel) -> BaseModel:
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=self.groq_model,
            response_format={"type": "json_object"},
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=30.0
        )
        response_text = chat_completion.choices[0].message.content
        return self._parse_json(response_text, response_model)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(settings.llm_retry_attempts))
    def _call_gemini(self, system_prompt: str, user_message: str, response_model: BaseModel) -> BaseModel:
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=[user_message],
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "temperature": settings.llm_temperature,
                "max_output_tokens": settings.llm_max_tokens,
            }
        )
        return self._parse_json(response.text, response_model)

    def _parse_json(self, text: str, model: Any) -> Any:
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(json_str)
            # Basic cleanup if model fields are expected to be strings but got lists
            if hasattr(model, 'model_fields'):
                for key, value in data.items():
                    if isinstance(value, list) and key in model.model_fields:
                        data[key] = " ".join([str(v) for v in value])
            return model(**data)
        except json.JSONDecodeError as e:
            raise LLMParseError(f"Invalid JSON from LLM: {str(e)}", text)
        except Exception as e:
            logger.error(f"Failed to parse LLM response as {model.__name__}", exc_info=True)
            raise LLMParseError(f"Schema validation failed: {str(e)}", text)

class DataManager:
    """Manage SQLite data operations using SQLModel"""
    def __init__(self, db_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    @staticmethod
    def _make_aware(dt: datetime) -> datetime:
        """Ensure a datetime is timezone-aware (UTC). Safe to call on already-aware datetimes."""
        if dt is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    def get_existing_video_ids(self) -> Set[str]:
        with Session(self.engine) as session:
            statement = select(InsightRecord.video_id)
            results = session.exec(statement).all()
            return set(results)

    def save_result(self, record: InsightRecord):
        with Session(self.engine) as session:
            session.add(record)
            session.commit()

    def get_channel_audit(self, channel_id: str) -> Optional[ChannelAudit]:
        with Session(self.engine) as session:
            audit = session.get(ChannelAudit, channel_id)
            if audit and audit.last_processed_at.tzinfo is None:
                audit.last_processed_at = DataManager._make_aware(audit.last_processed_at)
            return audit

    def update_channel_audit(self, audit: ChannelAudit):
        with Session(self.engine) as session:
            session.add(audit)
            session.commit()

class YouTubeService:
    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key, static_discovery=False)
        self.use_proxy = not settings.development_mode
        self.proxy_config = None
        if self.use_proxy and settings.webshare_username:
            self.proxy_config = WebshareProxyConfig(
                proxy_username=settings.webshare_username,
                proxy_password=settings.webshare_password,
            )
        # Instantiate once as requested, with proxy if configured
        self._transcript_api = YouTubeTranscriptApi(proxy_config=self.proxy_config)

    def get_latest_videos(self, channel_id: str, hours: int = 24) -> List[Dict]:
        try:
            published_after = (datetime.now() - timedelta(hours=hours)).isoformat() + 'Z'
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                publishedAfter=published_after,
                order='date',
                type='video',
                maxResults=settings.max_youtube_results
            )
            response = request.execute()
            videos = []
            for item in response.get('items', []):
                title = html.unescape(item['snippet']['title'])
                videos.append({
                    'video_id': item['id']['videoId'],
                    'title': title,
                    'channel_name': item['snippet']['channelTitle'],
                    'channel_id': channel_id
                })
            return videos
        except Exception as e:
            logger.error(f"Error fetching videos for channel {channel_id}", exc_info=True)
            raise e

    def get_transcript(self, video_id: str) -> Optional[str]:
        try:
            # Use the instance method 'list' which is available in this version
            transcript_list = self._transcript_api.list(video_id)
            
            transcript = transcript_list.find_transcript(['en'])
            pieces = transcript.fetch()
            return " ".join([p.text if hasattr(p, 'text') else p['text'] for p in pieces])
        except Exception as e:
            logger.warning(f"Failed to fetch transcript for {video_id}", exc_info=True)
            return None

class InsightWorkflow:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.compressor = SemanticTextCompressor()
        self.app = self._build_graph()

    def _extract_insights(self, state: WorkflowState) -> WorkflowState:
        node_name = "extract"
        start_time = time.perf_counter()
        if not state.get("compressed_transcript"):
            state["compressed_transcript"] = self.compressor.compress(state["transcript"])
        
        if not state["compressed_transcript"]:
            logger.warning(f"Blank compressed transcript for {state['video_id']}. Skipping LLM.")
            state["reject_reason"] = "Empty transcript"
            state["rejection_stage"] = "extraction"
            REJECTION_COUNT.labels(stage="extraction", reason="empty_transcript").inc()
            return state

        user_message = f"Channel: {state['channel_name']}\nVideo: {state['video_title']}\nTranscript: {state['compressed_transcript']}"
        try:
            state["insights"] = self.llm.call_llm(INSIGHT_EXTRACTION_SYSTEM_PROMPT, user_message, InsightExtraction)
            NODE_EXECUTION_COUNT.labels(node_name=node_name, status="success").inc()
        except LLMParseError as e:
            logger.error(f"Failed to extract insights for {state['video_id']}", exc_info=True)
            state["reject_reason"] = "LLM Parse Error"
            state["rejection_stage"] = "extraction"
            NODE_EXECUTION_COUNT.labels(node_name=node_name, status="error").inc()
        
        duration = time.perf_counter() - start_time
        NODE_DURATION.labels(node_name=node_name).observe(duration)
        logger.info(f"Insight extraction completed in {duration:.2f}s", extra={"video_id": state["video_id"]})
        return state

    def _bucket_filter(self, state: WorkflowState) -> WorkflowState:
        if state.get("rejection_stage"): return state
        insights = state["insights"]
        if insights.content_bucket == "REJECT":
            state["reject_reason"] = f"Bucket: {insights.extraction_notes or 'Rejected Topic'}"
            state["rejection_stage"] = "bucket_filter"
            REJECTION_COUNT.labels(stage="bucket_filter", reason="wrong_bucket").inc()
        elif insights.intended_signal == "none":
            state["reject_reason"] = "No clear signal"
            state["rejection_stage"] = "bucket_filter"
            REJECTION_COUNT.labels(stage="bucket_filter", reason="no_signal").inc()
        return state

    def _score_insights(self, state: WorkflowState) -> WorkflowState:
        if state.get("rejection_stage"): return state
        node_name = "score"
        start_time = time.perf_counter()
        
        scoring_prompt = INSIGHT_SCORING_SYSTEM_PROMPT
        user_message = f"Evaluate these insights: {state['insights'].model_dump_json()}"
        
        try:
            state["scores"] = self.llm.call_llm(scoring_prompt, user_message, InsightScores)
            NODE_EXECUTION_COUNT.labels(node_name=node_name, status="success").inc()
        except LLMParseError:
            state["reject_reason"] = "Scoring Parse Error"
            state["rejection_stage"] = "scoring"
            NODE_EXECUTION_COUNT.labels(node_name=node_name, status="error").inc()
            
        duration = time.perf_counter() - start_time
        NODE_DURATION.labels(node_name=node_name).observe(duration)
        logger.info(f"Insight scoring completed in {duration:.2f}s", extra={"video_id": state["video_id"]})
        return state

    def _decide_post_type(self, state: WorkflowState) -> WorkflowState:
        if state.get("rejection_stage"): return state
        i = state["insights"]
        if i.corrected_misconception: state["post_type"] = "learning_correction"
        elif i.mental_model: state["post_type"] = "framework"
        elif i.open_research_question: state["post_type"] = "research_question"
        else: state["post_type"] = "learning_insight"
        return state

    def _write_draft(self, state: WorkflowState) -> WorkflowState:
        if state.get("rejection_stage"): return state
        start_time = time.perf_counter()
        
        user_message = f"Type: {state['post_type']}\nContext: {state['video_title']}\nInsights: {state['insights'].model_dump_json()}"
        try:
            state["draft"] = self.llm.call_llm(LINKEDIN_DRAFT_SYSTEM_PROMPT, user_message, DraftPost)
        except LLMParseError:
            state["reject_reason"] = "Drafting Parse Error"
            state["rejection_stage"] = "drafting"

        duration = time.perf_counter() - start_time
        logger.info(f"Draft writing completed in {duration:.2f}s", extra={"video_id": state["video_id"]})
        return state

    def _final_approval(self, state: WorkflowState) -> WorkflowState:
        if state.get("rejection_stage"): return state
        scores = state['scores']
        conf = (scores.academic_depth_score + scores.non_obvious_score + scores.transferability_score) / 3
        if conf >= 0.75:
            state["approved_for_post"] = True
        else:
            state["approved_for_post"] = False
            state["reject_reason"] = f"Low confidence: {conf}"
            state["rejection_stage"] = "final_approval"
        state["processed_at"] = datetime.now().isoformat()
        return state

    def _handle_rejection(self, state: WorkflowState) -> WorkflowState:
        state["approved_for_post"] = False
        if not state.get("processed_at"): state["processed_at"] = datetime.now().isoformat()
        return state

    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("extract", self._extract_insights)
        workflow.add_node("bucket_filter", self._bucket_filter)
        workflow.add_node("score", self._score_insights)
        workflow.add_node("decide_post_type", self._decide_post_type)
        workflow.add_node("write", self._write_draft)
        workflow.add_node("final", self._final_approval)
        workflow.add_node("reject", self._handle_rejection)

        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "bucket_filter")

        def check_bucket(s):
            return "reject" if s.get("rejection_stage") else "continue"
        
        workflow.add_conditional_edges("bucket_filter", check_bucket, {"reject": "reject", "continue": "score"})

        def check_score(s):
            if s.get("rejection_stage"): return "reject"
            total = s["scores"].academic_depth_score + s["scores"].non_obvious_score + s["scores"].transferability_score
            return "reject" if total < settings.score_threshold else "continue"

        workflow.add_conditional_edges("score", check_score, {"reject": "reject", "continue": "decide_post_type"})
        
        workflow.add_edge("decide_post_type", "write")
        workflow.add_edge("write", "final")

        def check_final(s):
            return "approved" if s.get("approved_for_post") else "reject"
        
        workflow.add_conditional_edges("final", check_final, {"approved": END, "reject": "reject"})
        workflow.add_edge("reject", END)
        
        return workflow.compile()

    def process_video(self, video: dict) -> dict:
        initial_state = {
            "video_id": video["video_id"],
            "channel_id": video["channel_id"],
            "channel_name": video["channel_name"],
            "video_title": video["title"],
            "transcript": video["transcript"],
            "compressed_transcript": None,
            "insights": None,
            "scores": None,
            "post_type": None,
            "draft": None,
            "approved_for_post": False,
            "reject_reason": None,
            "rejection_stage": None,
            "processed_at": ""
        }
        final = self.app.invoke(initial_state)
        return final

class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.discord = Discord(url=webhook_url)

    def send_update(self, result: dict):
        try:
            if result.get("approved_for_post"):
                # Extract intended_signal from insights
                intended_signal = result.get("insights").intended_signal if result.get("insights") else "unknown"
                
                # Create emoji mapping for better visual representation
                signal_emoji = {
                    "professor": "🎓",
                    "recruiter": "💼",
                    "both": "🎓💼",
                    "none": "❓"
                }
                emoji = signal_emoji.get(intended_signal, "❓")
                draft_content = result['draft'].post_content if result.get('draft') else "N/A"
                msg = f"""🚀 **New AI Insight Ready!**
📺 **Title:** {result['video_title']}
🔗 **URL:** https://www.youtube.com/watch?v={result['video_id']}
{emoji} **Intended Signal:** {intended_signal}

**Draft Post:**
{draft_content}
"""
            else:
                msg = f"""🗑️ **Video Rejected**
📺 **Title:** {result['video_title']}
❌ **Reason:** {result['reject_reason']}
"""
            
            # Split if too long
            chunk_size = settings.discord_chunk_size
            for i in range(0, len(msg), chunk_size):
                self.discord.post(content=msg[i:i+chunk_size])
        except Exception:
            logger.error("Failed to send Discord notification", exc_info=True)

    def send_summary(self, total_processed: int, total_rejected: int, total_passed: int):
        """Send execution summary at the end of the run"""
        try:
            msg = f"""📊 **Daily Execution Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📹 **Total Videos Processed:** {total_processed}
✅ **Total Videos Passed:** {total_passed}
❌ **Total Videos Rejected:** {total_rejected}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            self.discord.post(content=msg)
        except Exception:
            logger.error("Failed to send Discord summary", exc_info=True)

class TranscriptProcessor:
    def __init__(self, dry_run: bool = False):
        self.config = settings
        self.data_manager = DataManager(settings.db_path)
        self.youtube = YouTubeService(settings.youtube_api_key)
        self.llm = LLMService(settings.groq_api_key, settings.gemini_api_key)
        self.workflow = InsightWorkflow(self.llm)
        self.notifier = DiscordNotifier(settings.discord_webhook_url)
        self.dry_run = dry_run

    def _flatten_result(self, final: dict) -> InsightRecord:
        return InsightRecord(
            video_id=final["video_id"],
            channel_id=final["channel_id"],
            channel_name=final["channel_name"],
            video_title=final["video_title"],
            transcript=final["transcript"],
            compressed_transcript=final["compressed_transcript"] or "",
            content_bucket=final["insights"].content_bucket if final.get("insights") else "REJECT",
            intended_signal=final["insights"].intended_signal if final.get("insights") else "none",
            post_type=final.get("post_type"),
            approved_for_post=final.get("approved_for_post", False),
            post_content=final["draft"].post_content if final.get("draft") else None,
            confidence_score=final["draft"].confidence_score if final.get("draft") else None,
            academic_depth=final["scores"].academic_depth_score if final.get("scores") else None,
            non_obvious=final["scores"].non_obvious_score if final.get("scores") else None,
            transferability=final["scores"].transferability_score if final.get("scores") else None,
            reject_reason=final.get("reject_reason"),
            rejection_stage=final.get("rejection_stage"),
            processed_at=datetime.now(UTC).replace(tzinfo=None)
        )

    async def process_channel(self, channel_id: str, existing_ids: Set[str]):
        # Circuit Breaker Check
        audit = self.data_manager.get_channel_audit(channel_id)
        if audit and audit.is_circuit_broken:
            logger.warning(f"Skipping channel {channel_id}: Circuit Broken.")
            return []

        logger.info(f"Checking channel: {channel_id}")
        try:
            # Use real lookback from audit if available, else settings
            lookback = settings.lookback_hours
            if audit:
                # Ensure we are comparing aware datetimes
                diff = datetime.now(UTC) - DataManager._make_aware(audit.last_processed_at)
                lookback = max(1, int(diff.total_seconds() / 3600) + 1)

            videos = self.youtube.get_latest_videos(channel_id, hours=lookback)
            new_videos = [v for v in videos if v['video_id'] not in existing_ids]
            
            # Reset audit on success
            if not audit:
                audit = ChannelAudit(channel_id=channel_id, last_processed_at=datetime.now(UTC).replace(tzinfo=None))
            else:
                audit.last_processed_at = datetime.now(UTC).replace(tzinfo=None)
                audit.consecutive_errors = 0
            self.data_manager.update_channel_audit(audit)
            
            return new_videos
        except Exception as e:
            logger.error(f"Error processing channel {channel_id}", exc_info=True)
            if not audit:
                audit = ChannelAudit(channel_id=channel_id, last_processed_at=datetime.now(UTC).replace(tzinfo=None), consecutive_errors=1)

            else:
                audit.consecutive_errors += 1
                if audit.consecutive_errors >= 3:
                    audit.is_circuit_broken = True
                    logger.critical(f"Circuit broken for channel {channel_id}")
            self.data_manager.update_channel_audit(audit)
            return []

    async def run(self):
        logger.info(f"Starting Transcript Processing Workflow v{__version__}")
        if self.dry_run:
            logger.info("DRY RUN MODE ENABLED - No writes or notifications will be sent.")
        
        try:
            with open(settings.channel_ids_file, 'r') as f:
                channel_ids = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Channel IDs file not found: {settings.channel_ids_file}")
            return

        existing_ids = self.data_manager.get_existing_video_ids()
        
        # Concurrent channel discovery
        tasks = [self.process_channel(cid, existing_ids) for cid in channel_ids]
        results = await asyncio.gather(*tasks)
        
        all_new_videos = [v for sublist in results for v in sublist]
        total_found = len(all_new_videos)
        total_processed = 0
        total_passed = 0
        total_rejected = 0

        # Process videos
        for v_info in all_new_videos:
            video_id = v_info['video_id']
            # Use Adapter for context logging
            v_logger = VideoContextAdapter(logger, {"video_id": video_id, "channel_id": v_info['channel_id']})
            v_logger.info(f"Processing video: {v_info['title']}")
            
            transcript = self.youtube.get_transcript(video_id)
            if not transcript:
                v_logger.warning("No transcript found, skipping.")
                continue
            
            v_info['transcript'] = transcript
            
            # Workflow is CPU/LLM bound, but involves sequential steps. 
            # For now, keeping it sequential per video to avoid rate limits, but using async for the overall structure.
            result_state = self.workflow.process_video(v_info)
            record = self._flatten_result(result_state)

            if not self.dry_run:
                self.data_manager.save_result(record)
                self.notifier.send_update(result_state)
            
            total_processed += 1
            
            # Track pass/reject counts
            if result_state.get("approved_for_post"):
                total_passed += 1
            else:
                total_rejected += 1
                
        logger.info(f"Batch complete. Found {total_found}, processed {total_processed}, passed {total_passed}, rejected {total_rejected}")
        
        # Send summary to Discord
        if not self.dry_run and total_processed > 0:
            self.notifier.send_summary(total_processed, total_rejected, total_passed)

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="YouTube to Discord Insight Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without writing to DB or Discord")
    args = parser.parse_args()

    try:
        processor = TranscriptProcessor(dry_run=args.dry_run)
        await processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())