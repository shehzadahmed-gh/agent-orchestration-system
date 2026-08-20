import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Unit tests in this directory exercise pure logic only (no live Postgres/
# Redis/Chroma needed) - CELERY_TASK_ALWAYS_EAGER and LLM_PROVIDER are set
# defensively in case a test imports something that reads settings.
os.environ.setdefault("LLM_PROVIDER", "local")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
