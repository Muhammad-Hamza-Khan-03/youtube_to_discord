from prometheus_client import Counter, Histogram, Summary

# Node execution counters
NODE_EXECUTION_COUNT = Counter(
    "workflow_node_execution_total",
    "Total number of times a workflow node was executed",
    ["node_name", "status"]
)

# Insight scoring results
INSIGHT_SCORE = Summary(
    "insight_score_total",
    "Total scores assigned to insights",
    ["video_id"]
)

# Processing time
NODE_DURATION = Histogram(
    "workflow_node_duration_seconds",
    "Time spent in each workflow node",
    ["node_name"]
)

# Reject rate
REJECTION_COUNT = Counter(
    "workflow_rejection_total",
    "Total number of rejections",
    ["stage", "reason"]
)
