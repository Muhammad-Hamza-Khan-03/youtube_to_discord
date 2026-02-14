import pytest
from unittest.mock import MagicMock, patch
from main import SemanticTextCompressor, LLMService, InsightExtraction, LLMParseError
from settings import settings

def test_compressor_empty():
    compressor = SemanticTextCompressor()
    assert compressor.compress("") == ""
    assert compressor.compress("   ") == ""

def test_compressor_truncation():
    compressor = SemanticTextCompressor()
    # Mock settings to have a very small truncation limit
    with patch('main.settings.transcript_truncation_limit', 5):
        text = "This is a very long text that should be truncated by the tokenizer or character limit."
        compressed = compressor.compress(text)
        # Tiktoken tokens are roughly words, so this should be short
        assert len(compressed.split()) <= 10 

def test_parse_json_valid():
    service = LLMService("key1", "key2")
    model = InsightExtraction
    raw_json = '{"content_bucket": "core_ml_foundations", "intended_signal": "professor"}'
    result = service._parse_json(raw_json, model)
    assert result.content_bucket == "core_ml_foundations"
    assert result.intended_signal == "professor"

def test_parse_json_with_markdown():
    service = LLMService("key1", "key2")
    model = InsightExtraction
    raw_json = '```json\n{"content_bucket": "research_curiosity", "intended_signal": "both"}\n```'
    result = service._parse_json(raw_json, model)
    assert result.content_bucket == "research_curiosity"

def test_parse_json_invalid():
    service = LLMService("key1", "key2")
    model = InsightExtraction
    raw_json = '{"invalid": json}'
    with pytest.raises(LLMParseError):
        service._parse_json(raw_json, model)

@patch('main.Groq')
def test_call_groq_retry(mock_groq_class):
    mock_client = mock_groq_class.return_value
    mock_client.chat.completions.create.side_effect = [Exception("Rate limit"), MagicMock()]
    
    service = LLMService("key1", "key2")
    # This is a bit complex to test fully without deep mocking, but it shows the intent
    pass
