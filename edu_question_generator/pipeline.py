"""Full pipeline: document → segments → MCQ generation → merge → validate → select.

Strategy:
1. Extract text
2. Logical split (headings) capped at MAX_LOGICAL_SEGMENTS
3. Distribute question budget across segments
4. Merge, dedupe, validate, then cap to TARGET_QUESTIONS_TOTAL
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .chunking import build_logical_segments
from .config import (
    DEEPSEEK_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_INVALID_MODEL,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
    LOGICAL_SEGMENT_MAX_CHARS,
    MAX_LOGICAL_SEGMENTS,
    PIPELINE_ALL_SEGMENTS_FAILED,
    TARGET_COMPUTATION_MIN,
    TARGET_ANALYSIS_APPLICATION_MIN,
    TARGET_QUESTIONS_TOTAL,
)
from .generator import Lang, generate_questions
from .question_selection import (
    cap_and_diversify_mcq,
    dedupe_mcq,
    distribute_question_counts,
)
from .validator import filter_mcq_payload

# Progress callback: (stage_name, progress_data)
PipelineProgressCallback = Callable[[str, dict[str, Any]], None]


def _resolve_model_id(model: str) -> str:
    """Resolve model identifier (from ui/secrets or config default)."""
    return model or DEEPSEEK_MODEL


def _emit_progress(
    callback: PipelineProgressCallback | None,
    stage: str,
    **payload: Any,
) -> None:
    """Notify the UI of a pipeline stage."""
    if callback is not None:
        callback(stage, payload)


def merge_payloads(payloads: list[dict]) -> dict:
    """Merge mcq lists from all segments then deduplicate."""
    merged: list[dict] = []
    for payload in payloads:
        merged.extend(payload.get("mcq", []))
    return {"mcq": dedupe_mcq(merged)}


# Errors that abort the pipeline immediately (not treated as segment_skip)
_FATAL_LLM_ERRORS = frozenset({
    LLM_LIMIT_ERROR,
    LLM_INSUFFICIENT_BALANCE,
    LLM_INVALID_MODEL,
})


def _generate_segment_payload(
    segment: str,
    lang: Lang,
    num_questions: int | None,
    model: str,
    api_key: str | None,
) -> dict | None:
    """Generate MCQs from one segment; None on parse failure or oversized request."""
    if num_questions == 0:
        return {"mcq": []}
    try:
        return generate_questions(
            segment,
            lang,
            num_questions,
            model,
            api_key,
        )
    except json.JSONDecodeError:
        return None
    except RuntimeError as exc:
        message = str(exc)
        if message == LLM_REQUEST_TOO_LARGE:
            return None
        if message in _FATAL_LLM_ERRORS:
            raise
        return None


def generate_from_document(
    text: str,
    lang: Lang,
    num_questions: int | None = None,
    model: str = "",
    api_key: str | None = None,
    *,
    progress_callback: PipelineProgressCallback | None = None,
    target_questions: int | None = None,
    target_computation_min: int | None = None,
) -> tuple[dict, dict]:
    """Run the full pipeline: split → generate → merge → validate → select.

    Args:
        text: Full document text.
        lang: Document language (ar/en).
        num_questions: Question budget (legacy; prefer target_questions).
        model: DeepSeek model identifier.
        api_key: API key.
        progress_callback: (stage, data) function for UI updates.
        target_questions: Final count target (default TARGET_QUESTIONS_TOTAL).
        target_computation_min: Minimum computation questions.

    Returns:
        (final payload, run metadata).

    Raises:
        RuntimeError: PIPELINE_ALL_SEGMENTS_FAILED or critical LLM errors.
    """
    question_budget = (
        target_questions
        if target_questions is not None
        else (num_questions if num_questions is not None else TARGET_QUESTIONS_TOTAL)
    )
    computation_goal = (
        target_computation_min
        if target_computation_min is not None
        else TARGET_COMPUTATION_MIN
    )

    segments = build_logical_segments(
        text,
        max_segments=MAX_LOGICAL_SEGMENTS,
        max_segment_chars=LOGICAL_SEGMENT_MAX_CHARS,
    )

    if not segments:
        return {"mcq": []}, {
            "text_chars": len(text),
            "segments_total": 0,
            "segments_used": 0,
            "segments_skipped": 0,
            "segment_max_chars": LOGICAL_SEGMENT_MAX_CHARS,
            "target_questions": question_budget,
            "segmentation_mode": "logical",
        }

    segments_total = len(segments)

    _emit_progress(
        progress_callback,
        "chunking",
        text_chars=len(text),
        segments_total=segments_total,
        segments_used=len(segments),
        segment_max_chars=LOGICAL_SEGMENT_MAX_CHARS,
        target_questions=question_budget,
    )

    payloads: list[dict] = []
    segments_skipped = 0
    model_id = _resolve_model_id(model)

    per_segment_counts = distribute_question_counts(question_budget, len(segments))
    segment_jobs = list(zip(segments, per_segment_counts))

    total_jobs = len(segment_jobs)

    for job_index, (segment, segment_count) in enumerate(segment_jobs, start=1):
        if segment_count <= 0:
            continue
        _emit_progress(
            progress_callback,
            "segment_llm_start",
            index=job_index,
            total=total_jobs,
            model=model_id,
            segment_chars=len(segment),
            segment_questions=segment_count,
        )
        payload = _generate_segment_payload(
            segment,
            lang,
            segment_count,
            model,
            api_key,
        )
        if payload is not None:
            payloads.append(payload)
            _emit_progress(
                progress_callback,
                "segment_llm_done",
                index=job_index,
                total=total_jobs,
                model=model_id,
                mcq_count=len(payload.get("mcq", [])),
            )
        else:
            segments_skipped += 1
            _emit_progress(
                progress_callback,
                "segment_skip",
                index=job_index,
                total=total_jobs,
                model=model_id,
            )

    if not payloads:
        raise RuntimeError(PIPELINE_ALL_SEGMENTS_FAILED)

    mcq_before_merge = sum(len(p.get("mcq", [])) for p in payloads)
    _emit_progress(
        progress_callback,
        "merge",
        mcq_raw=mcq_before_merge,
        segment_payloads=len(payloads),
    )
    merged = merge_payloads(payloads)
    mcq_after_dedupe = len(merged.get("mcq", []))
    _emit_progress(
        progress_callback,
        "merge_done",
        mcq_after_dedupe=mcq_after_dedupe,
    )
    _emit_progress(progress_callback, "validate", mcq_before_filter=mcq_after_dedupe)
    meta = {
        "text_chars": len(text),
        "segments_total": segments_total,
        "segments_used": len(segments),
        "segments_skipped": segments_skipped,
        "segment_max_chars": LOGICAL_SEGMENT_MAX_CHARS,
        "segmentation_mode": "logical",
        "target_questions": question_budget,
        "target_computation_min": computation_goal,
        "target_analysis_application_min": TARGET_ANALYSIS_APPLICATION_MIN,
        "model_used": model_id,
    }
    filtered = filter_mcq_payload(
        merged,
        text,
        api_key,
        model,
    )
    mcq_before_cap = len(filtered.get("mcq", []))
    _emit_progress(
        progress_callback,
        "cap",
        mcq_before_cap=mcq_before_cap,
        target_questions=question_budget,
    )
    capped = {
        "mcq": cap_and_diversify_mcq(
            list(filtered.get("mcq", [])),
            max_total=question_budget,
            min_computation=computation_goal,
            min_analysis_application=TARGET_ANALYSIS_APPLICATION_MIN,
        ),
    }
    mcq_final = len(capped.get("mcq", []))
    meta["mcq_before_cap"] = mcq_before_cap
    meta["mcq_final"] = mcq_final
    _emit_progress(
        progress_callback,
        "done",
        mcq_final=mcq_final,
        model_used=model_id,
        segments_skipped=segments_skipped,
    )
    return capped, meta
