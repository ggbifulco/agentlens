"""LLM-as-judge machinery (``docs/design.md`` §5, FR-E2/FR-E4).

The reusable parts behind the ``llm_rubric`` and ``llm_pass_fail`` evaluators
(:mod:`agentlens.evals.builtin.judges`) and the pairwise judge:

* :class:`BaseJudge` — client, cache and call flow shared by every judge;
* :class:`PairwiseJudge` — compares two traces of a case in both orders
  (:mod:`~agentlens.evals.judge.pairwise`);
* :class:`LLMJudge` — abstract evaluator: client resolution, ``temperature=0``
  JSON-mode call, robust parsing with retry, caching, sync and async;
* :mod:`~agentlens.evals.judge.prompts` — packaged ``string.Template`` prompts;
* :mod:`~agentlens.evals.judge.parse` — finding and validating the JSON verdict;
* :mod:`~agentlens.evals.judge.cache` — :class:`JudgeCache` with in-memory and
  on-disk implementations keyed by :func:`judge_cache_key`.
"""

from agentlens.evals.judge.base import (
    JSON_RESPONSE_FORMAT,
    JUDGE_MODEL_ENV,
    JUDGE_TEMPERATURE,
    MAX_RETRIES,
    BaseJudge,
    JudgePrompt,
    JudgeRun,
    LLMJudge,
    value_text,
)
from agentlens.evals.judge.cache import (
    DEFAULT_JUDGE_CACHE_DIR,
    JUDGE_CACHE_ENV,
    CacheSpec,
    DiskJudgeCache,
    JudgeCache,
    MemoryJudgeCache,
    judge_cache_key,
    resolve_cache,
    shared_memory_cache,
)
from agentlens.evals.judge.pairwise import (
    ORDERS,
    PAIRWISE_NAME,
    PAIRWISE_PLACEHOLDERS,
    Order,
    OrderVerdict,
    PairwiseJudge,
    PairwiseResult,
    PairwiseVerdict,
    Position,
    Winner,
    parse_position,
)
from agentlens.evals.judge.parse import (
    JudgeParseError,
    extract_json_object,
    parse_reasoning,
    parse_score,
    parse_verdict,
)
from agentlens.evals.judge.prompts import check_template, load_template, render_template

__all__ = [
    "DEFAULT_JUDGE_CACHE_DIR",
    "JSON_RESPONSE_FORMAT",
    "JUDGE_CACHE_ENV",
    "JUDGE_MODEL_ENV",
    "JUDGE_TEMPERATURE",
    "MAX_RETRIES",
    "ORDERS",
    "PAIRWISE_NAME",
    "PAIRWISE_PLACEHOLDERS",
    "BaseJudge",
    "CacheSpec",
    "DiskJudgeCache",
    "JudgeCache",
    "JudgeParseError",
    "JudgePrompt",
    "JudgeRun",
    "LLMJudge",
    "MemoryJudgeCache",
    "Order",
    "OrderVerdict",
    "PairwiseJudge",
    "PairwiseResult",
    "PairwiseVerdict",
    "Position",
    "Winner",
    "check_template",
    "extract_json_object",
    "judge_cache_key",
    "load_template",
    "parse_position",
    "parse_reasoning",
    "parse_score",
    "parse_verdict",
    "render_template",
    "resolve_cache",
    "shared_memory_cache",
    "value_text",
]
