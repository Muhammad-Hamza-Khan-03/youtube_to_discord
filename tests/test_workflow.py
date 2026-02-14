import pytest
from unittest.mock import MagicMock
from main import InsightWorkflow, WorkflowState, InsightExtraction, InsightScores, DraftPost

@pytest.fixture
def mock_llm():
    service = MagicMock()
    # Mocking extraction
    service.call_llm.side_effect = [
        InsightExtraction(
            content_bucket="core_ml_foundations",
            intended_signal="professor",
            non_obvious_insight="Deep learning is actually just math."
        ),
        # Mocking scores
        InsightScores(
            academic_depth_score=0.9,
            non_obvious_score=0.9,
            transferability_score=0.9
        ),
        # Mocking draft
        DraftPost(
            post_content="Met many deep learners today...",
            confidence_score=0.9
        )
    ]
    return service

def test_workflow_full_run(mock_llm):
    workflow = InsightWorkflow(mock_llm)
    video_info = {
        "video_id": "test_id",
        "channel_id": "chan_id",
        "channel_name": "DeepMind",
        "title": "AlphaGo Documentary",
        "transcript": "In this video we talk about reinforcement learning and its foundations."
    }
    
    result = workflow.process_video(video_info)
    assert result["approved_for_post"] is True
    assert result["video_id"] == "test_id"
    assert result["draft"].post_content == "Met many deep learners today..."
