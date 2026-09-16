"""BoundGrounder: Domain-anchored generation verification mechanism (Innovation Point 3)

Four-step verification pipeline:
  1. Source Tracing
  2. Plausibility Check
  3. Multi-source Consistency Check
  4. Confidence Synthesis
"""

import re
from dataclasses import dataclass, field

from src.generator.base import BaseGenerator, GenerationResult
from src.retriever.base import RetrievalResult
from src.config import DEFAULT_CONFIG
from src.utils.constraint_taxonomy import (
    PHYSICAL_BOUNDS,
    PARAM_TO_BOUND,
    PhysicalBound,
    EquipmentProfile,
    EQUIPMENT_PROFILES,
)
from src.utils.logger import get_logger
from src.utils.llm_client import get_llm_client

logger = get_logger(__name__)


@dataclass
class ConstraintValue:
    """Constraint value extracted from generated text"""
    parameter: str
    value: float
    unit: str
    source_chunk_id: str | None = None
    source_quote: str = ""
    source_score: float = 0.0  # source tracing score
    plausibility_score: float = 0.0  # physical plausibility score
    consistency_score: float = 0.0  # multi-source consistency score
    confidence: float = 0.0  # aggregated confidence
    is_hallucination: bool = False  # whether flagged as suspected hallucination
    flags: list[str] = field(default_factory=list)  # warning flags


class BoundGrounder(BaseGenerator):
    """Domain-anchored generation verification"""

    name = "BoundGrounder"

    # Numeric value + unit extraction pattern
    NUMERIC_PATTERN = re.compile(
        r"(\w+)\s*[：:=]?\s*[≥≤≮≯><]?\s*(\d+\.?\d*)\s*(mm|cm|m|kg|N|h|%|°|步|小时)"
    )

    def __init__(self, llm_client=None, equipment_type: str | None = None):
        cfg = DEFAULT_CONFIG.generator
        self.llm = llm_client or get_llm_client()
        self.w_source = cfg.w_source
        self.w_plausibility = cfg.w_plausibility
        self.w_consistency = cfg.w_consistency
        self.confidence_threshold = cfg.confidence_threshold

        self.equipment_profile: EquipmentProfile | None = None
        if equipment_type and equipment_type in EQUIPMENT_PROFILES:
            self.equipment_profile = EQUIPMENT_PROFILES[equipment_type]

    def generate(self, query: str, retrieved: list[RetrievalResult]) -> GenerationResult:
        """Generate and verify"""

        # Step 0: Build context and generate initial answer
        context = self._build_context(retrieved)
        raw_answer = self._generate_grounded(query, context)

        # Step 1: Source tracing
        constraint_values = self._extract_constraint_values(raw_answer)
        self._trace_sources(constraint_values, retrieved)

        # Step 2: Physical plausibility check
        self._check_plausibility(constraint_values)

        # Step 3: Multi-source consistency check
        self._check_consistency(constraint_values, retrieved)

        # Step 4: Confidence synthesis
        self._synthesize_confidence(constraint_values)

        # Build final answer with verification metadata
        final_answer = self._build_verified_answer(raw_answer, constraint_values)
        sources = [r.chunk.chunk_id for r in retrieved]

        overall_confidence = (
            sum(cv.confidence for cv in constraint_values) / len(constraint_values)
            if constraint_values else 0.8
        )

        return GenerationResult(
            answer=final_answer,
            sources=sources,
            confidence=overall_confidence,
            metadata={
                "constraint_values": [
                    {
                        "parameter": cv.parameter,
                        "value": cv.value,
                        "unit": cv.unit,
                        "confidence": cv.confidence,
                        "is_hallucination": cv.is_hallucination,
                        "flags": cv.flags,
                    }
                    for cv in constraint_values
                ],
                "method": "bound_grounder",
            },
        )

    def _generate_grounded(self, query: str, context: str) -> str:
        """Generate answer with source annotations"""
        prompt = f"""You are a maintainability engineering expert. Answer the question strictly based on the reference materials.

Requirements:
1. Every constraint value must cite its source: [value] (source: chunk_id, original: "quoted text")
2. Mark unsupported numerical values as [low_confidence]
3. Do not fabricate any numerical values

Reference Materials:
{context}

Question: {query}

Answer:"""
        return self.llm.generate(prompt)

    def _extract_constraint_values(self, text: str) -> list[ConstraintValue]:
        """Extract constraint values from generated text"""
        values = []
        for match in self.NUMERIC_PATTERN.finditer(text):
            param, val, unit = match.groups()
            values.append(ConstraintValue(
                parameter=param,
                value=float(val),
                unit=unit,
            ))
        return values

    def _trace_sources(self, values: list[ConstraintValue], retrieved: list[RetrievalResult]):
        """Step 1: Source tracing — verify that generated values exist in retrieval results"""
        for cv in values:
            val_str = str(cv.value)
            if val_str.endswith(".0"):
                val_str = val_str[:-2]

            found = False
            for r in retrieved:
                if val_str in r.chunk.content:
                    cv.source_chunk_id = r.chunk.chunk_id
                    cv.source_score = 1.0
                    # Extract surrounding context from source
                    idx = r.chunk.content.index(val_str)
                    start = max(0, idx - 30)
                    end = min(len(r.chunk.content), idx + len(val_str) + 30)
                    cv.source_quote = r.chunk.content[start:end]
                    found = True
                    break

            if not found:
                cv.source_score = 0.3  # lenient strategy: not found does not zero out, allows reasonable inference
                cv.flags.append("inferred: value not directly found in retrieval results, may be reasonable inference")

    def _check_plausibility(self, values: list[ConstraintValue]):
        """Step 2: Physical plausibility check — whether the value falls within a reasonable range"""
        for cv in values:
            bound = self._find_matching_bound(cv)
            if bound is None:
                cv.plausibility_score = 0.8  # unknown parameter gets high score to avoid false positives
                continue

            # Use equipment-specific bounds if available
            lower, upper = bound.lower, bound.upper
            if self.equipment_profile:
                custom = self.equipment_profile.custom_bounds.get(bound.parameter)
                if custom:
                    lower, upper = custom

            if lower <= cv.value <= upper:
                cv.plausibility_score = 1.0
            else:
                cv.plausibility_score = 0.0
                cv.is_hallucination = True
                cv.flags.append(
                    f"out_of_bounds: {cv.value}{cv.unit} exceeds reasonable range [{lower}, {upper}]{bound.unit}"
                )

    def _check_consistency(self, values: list[ConstraintValue], retrieved: list[RetrievalResult]):
        """Step 3: Multi-source consistency check — whether multiple sources give consistent values"""
        for cv in values:
            val_str = str(cv.value)
            if val_str.endswith(".0"):
                val_str = val_str[:-2]

            # Count sources mentioning the same parameter and those with matching values
            source_count = 0
            agree_count = 0

            for r in retrieved:
                content = r.chunk.content
                # Check if this source mentions a similar parameter
                if cv.parameter in content or cv.unit in content:
                    source_count += 1
                    if val_str in content:
                        agree_count += 1

            if source_count > 0:
                cv.consistency_score = agree_count / source_count
                if cv.consistency_score < 0.5 and source_count >= 2:
                    cv.flags.append(
                        f"inconsistent: {agree_count}/{source_count} sources agree"
                    )
            else:
                cv.consistency_score = 0.5  # cannot determine

    def _synthesize_confidence(self, values: list[ConstraintValue]):
        """Step 4: Confidence synthesis"""
        for cv in values:
            cv.confidence = (
                self.w_source * cv.source_score
                + self.w_plausibility * cv.plausibility_score
                + self.w_consistency * cv.consistency_score
            )
            if cv.confidence < self.confidence_threshold:
                cv.flags.append(f"low_confidence: {cv.confidence:.2f}")

    def _build_verified_answer(self, raw_answer: str, values: list[ConstraintValue]) -> str:
        """Build final answer with verification metadata (only flag clear hallucinations, do not reject due to low confidence)"""
        warnings = []
        for cv in values:
            if cv.is_hallucination:  # only flag clear hallucinations that violate physical bounds
                warnings.append(
                    f"[Suspected Hallucination] {cv.parameter}={cv.value}{cv.unit} — {'; '.join(cv.flags)}"
                )
            # Removed low-confidence warnings to avoid excessive refusal

        if warnings:
            return raw_answer + "\n\n---\nVerification Warnings:\n" + "\n".join(warnings)
        return raw_answer

    def _find_matching_bound(self, cv: ConstraintValue) -> PhysicalBound | None:
        """Find the matching physical bound"""
        # Exact match
        if cv.parameter in PARAM_TO_BOUND:
            return PARAM_TO_BOUND[cv.parameter]
        # Fuzzy match
        for bound in PHYSICAL_BOUNDS:
            if bound.unit == cv.unit and (
                bound.parameter.lower() in cv.parameter.lower()
                or cv.parameter.lower() in bound.parameter.lower()
            ):
                return bound
        return None

