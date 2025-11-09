"""
meta_selector.py - Updated Meta-Selector Implementation

Based on revised theoretical framework with:
1. Post-hoc validation of selector decisions using actual outputs
2. Explicit escalation rules with Expected Value of Information (EVI)
3. Session state tracking and intent repetition detection
4. Confidence calibration and hysteresis
5. Budget management and criticality handling

Author: Updated by Claude based on revised framework
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import logging

# Import from updated selector
from selector_v2 import (
    BehavioralSelector,
    ModelAdapter,
    EmbeddingSpace,
    SelectionResult,
    SelectionMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Session State Management
# ============================================================================

@dataclass
class SessionState:
    """
    Tracks session-level context for meta-selection decisions.
    
    Key responsibilities:
    - Track query history and detect repetition (indicates user dissatisfaction)
    - Monitor confidence trends over time
    - Manage computational budget
    - Record model usage patterns
    """
    
    # Query history
    query_history: List[str] = field(default_factory=list)
    query_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Performance tracking
    confidence_history: List[float] = field(default_factory=list)
    distance_history: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)
    
    # Model usage
    model_history: List[str] = field(default_factory=list)
    level_history: List[int] = field(default_factory=list)
    
    # Budget and criticality
    initial_budget: float = 1.0
    budget_spent: float = 0.0
    is_critical: bool = False
    
    # Intent tracking
    current_repetition_count: int = 0
    prev_query_embedding: Optional[np.ndarray] = None
    repetition_similarity_threshold: float = 0.9
    
    @property
    def budget_remaining(self) -> float:
        """Remaining computational budget."""
        return self.initial_budget - self.budget_spent
    
    @property
    def total_interactions(self) -> int:
        """Total number of interactions so far."""
        return len(self.query_history)
    
    def add_interaction(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: SelectionResult,
    ):
        """Record a completed interaction."""
        self.query_history.append(query)
        self.query_embeddings.append(query_embedding)
        self.model_history.append(result.chosen_model_id)
        self.level_history.append(result.chosen_level)
        self.confidence_history.append(result.confidence)
        self.distance_history.append(result.behavioral_distance)
        
        if result.prediction_error is not None:
            self.prediction_errors.append(result.prediction_error)
        
        self.budget_spent += result.estimated_cost
    
    def update_intent_repetition(
        self,
        current_query_embedding: np.ndarray
    ):
        """
        Detect if user is repeating similar queries.
        
        High repetition count indicates user dissatisfaction with responses,
        suggesting need for escalation to higher-quality model.
        
        Uses cosine similarity between current and previous query embeddings.
        """
        if self.prev_query_embedding is None:
            self.current_repetition_count = 1
        else:
            # Compute cosine similarity
            dot = np.dot(current_query_embedding, self.prev_query_embedding)
            norm_curr = np.linalg.norm(current_query_embedding)
            norm_prev = np.linalg.norm(self.prev_query_embedding)
            similarity = dot / (norm_curr * norm_prev + 1e-12)
            
            if similarity >= self.repetition_similarity_threshold:
                self.current_repetition_count += 1
                logger.info(
                    f"Intent repetition detected: count={self.current_repetition_count}, "
                    f"similarity={similarity:.3f}"
                )
            else:
                self.current_repetition_count = 1
        
        self.prev_query_embedding = current_query_embedding.copy()
    
    def get_confidence_trend(self, window: int = 3) -> str:
        """
        Analyze recent confidence trend.
        
        Returns: "consistently_high", "consistently_low", "declining", "improving", "mixed"
        """
        if len(self.confidence_history) < window:
            return "insufficient_data"
        
        recent = self.confidence_history[-window:]
        
        if all(c >= 0.75 for c in recent):
            return "consistently_high"
        elif all(c < 0.5 for c in recent):
            return "consistently_low"
        elif recent[-1] < recent[0] - 0.1:
            return "declining"
        elif recent[-1] > recent[0] + 0.1:
            return "improving"
        else:
            return "mixed"
    
    def get_average_prediction_error(self, window: int = 5) -> float:
        """Get average prediction error over recent window."""
        if not self.prediction_errors:
            return 0.0
        recent = self.prediction_errors[-window:]
        return float(np.mean(recent))


# ============================================================================
# Escalation Decision Types
# ============================================================================

class EscalationReason(Enum):
    """Reasons for escalating to a higher-level model."""
    NO_ESCALATION = "no_escalation"
    LOW_CONFIDENCE = "low_confidence"
    INTENT_REPETITION = "intent_repetition"
    CRITICAL_TASK = "critical_task"
    POSITIVE_EVI = "positive_evi"  # Expected Value of Information > 0
    HIGH_PREDICTION_ERROR = "high_prediction_error"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class EscalationDecision:
    """
    Result of a meta-selector escalation decision.
    
    Contains both the decision (escalate or not) and the reasoning.
    """
    should_escalate: bool
    reason: EscalationReason
    
    # Target (if escalating)
    target_level: int
    target_model_id: Optional[str]
    
    # Justification
    current_confidence: float
    expected_improvement: float  # ΔQ
    expected_additional_cost: float  # ΔC
    evi_score: float  # ΔQ - λ×ΔC
    
    # Context
    decision_confidence: float  # How confident are we in this escalation decision?


# ============================================================================
# The Meta-Selector
# ============================================================================

class MetaSelector:
    """
    Meta-selector for validating and escalating selector decisions.
    
    THEORY:
    The meta-selector observes actual outcomes and decides whether escalation
    to a higher-capacity model is warranted based on:
    1. Observed confidence (from actual behavioral distance)
    2. Session state (repetition, budget, criticality)
    3. Expected Value of Information (rational cost-benefit)
    
    KEY RESPONSIBILITIES:
    1. Post-hoc validation: Measure actual behavioral distance
    2. Confidence monitoring: Track if selector predictions are reliable
    3. Escalation management: Decide when to invoke higher-level models
    4. Budget management: Prevent runaway costs
    5. Intent detection: Recognize user dissatisfaction
    
    ESCALATION RULES (in priority order):
    1. Budget exhausted → No escalation possible
    2. Critical task + low confidence → Escalate immediately
    3. Intent repetition + low confidence → Escalate
    4. Positive EVI (ΔQ - λ×ΔC > 0) → Rational escalation
    5. General low confidence → Escalate if budget allows
    
    HYSTERESIS:
    Once escalated, maintain higher level until confidence stabilizes.
    Prevents rapid oscillation between levels.
    """
    
    def __init__(
        self,
        selector: BehavioralSelector,
        embedding_space: EmbeddingSpace,
        model_adapters: Dict[str, ModelAdapter],
        # Confidence thresholds
        confidence_threshold: float = 0.75,
        critical_confidence_threshold: float = 0.65,
        # Intent repetition
        intent_repetition_threshold: int = 2,
        # EVI parameters
        lambda_cost: float = 0.6,  # Cost weight in EVI
        quality_improvement_rate: float = 0.3,  # Heuristic: +30% per level
        # Hysteresis
        hysteresis_margin: float = 0.05,
        hysteresis_window: int = 3,
        # Prediction error
        high_prediction_error_threshold: float = 0.2,
    ):
        self.selector = selector
        self.space = embedding_space
        self.models = model_adapters
        
        self.confidence_threshold = confidence_threshold
        self.critical_confidence_threshold = critical_confidence_threshold
        self.intent_repetition_threshold = intent_repetition_threshold
        self.lambda_cost = lambda_cost
        self.quality_improvement_rate = quality_improvement_rate
        self.hysteresis_margin = hysteresis_margin
        self.hysteresis_window = hysteresis_window
        self.high_prediction_error_threshold = high_prediction_error_threshold
        
        # Organize models by level for escalation
        self.levels = sorted(set(a.level for a in model_adapters.values()))
        self.models_by_level = self._organize_models_by_level()
    
    def _organize_models_by_level(self) -> Dict[int, List[str]]:
        """Organize models by hierarchy level."""
        by_level = {}
        for model_id, adapter in self.models.items():
            if adapter.level not in by_level:
                by_level[adapter.level] = []
            by_level[adapter.level].append(model_id)
        return by_level
    
    def _get_next_level_model(self, current_level: int) -> Optional[Tuple[int, str]]:
        """
        Get a model from the next higher level.
        
        Returns: (next_level, model_id) or None if at max level
        """
        next_level = current_level + 1
        if next_level in self.models_by_level and self.models_by_level[next_level]:
            # For simplicity, choose first model at next level
            # Could implement more sophisticated intra-level selection
            return next_level, self.models_by_level[next_level][0]
        return None
    
    def _estimate_quality_improvement(
        self,
        current_distance: float,
        current_level: int,
        target_level: int
    ) -> float:
        """
        Estimate expected improvement in behavioral distance from escalation.
        
        Heuristic: Each level improves distance by quality_improvement_rate.
        
        Formula:
        ΔQ = current_distance × (1 - (1 - r)^Δlevel)
        
        where r = quality_improvement_rate, Δlevel = target - current
        
        This is a simplification; in practice, could learn from historical data.
        """
        if target_level <= current_level:
            return 0.0
        
        level_diff = target_level - current_level
        expected_improvement = current_distance * (
            1.0 - (1.0 - self.quality_improvement_rate) ** level_diff
        )
        
        return expected_improvement
    
    def _compute_evi(
        self,
        current_distance: float,
        current_level: int,
        target_level: int,
        target_model_id: str,
        query: str
    ) -> float:
        """
        Compute Expected Value of Information for escalation.
        
        EVI = ΔQ - λ × ΔC
        
        where:
        - ΔQ = expected quality improvement (reduction in behavioral distance)
        - ΔC = additional computational cost
        - λ = cost weight parameter
        
        Escalate if EVI > 0 (expected benefit exceeds weighted cost).
        """
        # Expected quality improvement
        delta_quality = self._estimate_quality_improvement(
            current_distance, current_level, target_level
        )
        
        # Additional cost
        delta_cost = self.models[target_model_id].estimate_cost(query)
        
        # EVI
        evi = delta_quality - self.lambda_cost * delta_cost
        
        logger.debug(
            f"EVI computation: ΔQ={delta_quality:.4f}, ΔC={delta_cost:.6f}, "
            f"λ={self.lambda_cost}, EVI={evi:.4f}"
        )
        
        return evi
    
    def validate_and_decide(
        self,
        query: str,
        current_result: SelectionResult,
        state: SessionState,
        force_critical: bool = False
    ) -> EscalationDecision:
        """
        Main meta-selector decision: validate current result and decide escalation.
        
        This implements the complete escalation logic with priority rules.
        
        Algorithm:
        1. Update session state with current result
        2. Check budget constraints
        3. Check if already at max level
        4. Apply escalation rules in priority order:
           a. Critical task + low confidence
           b. Intent repetition + low confidence
           c. High prediction error (selector miscalibrated)
           d. Positive EVI
           e. General low confidence
        5. Apply hysteresis if needed
        6. Return decision
        
        Returns:
            EscalationDecision with reasoning and target
        """
        # Update session state
        query_embedding = self.space.embed_query(query)
        state.update_intent_repetition(query_embedding)
        state.add_interaction(query, query_embedding, current_result)
        
        if force_critical:
            state.is_critical = True
        
        current_level = current_result.chosen_level
        confidence = current_result.confidence
        distance = current_result.behavioral_distance
        
        # Check budget
        if state.budget_remaining <= 0:
            logger.warning("Budget exhausted, cannot escalate")
            return EscalationDecision(
                should_escalate=False,
                reason=EscalationReason.NO_ESCALATION,
                target_level=current_level,
                target_model_id=current_result.chosen_model_id,
                current_confidence=confidence,
                expected_improvement=0.0,
                expected_additional_cost=0.0,
                evi_score=0.0,
                decision_confidence=1.0
            )
        
        # Check if at max level
        next_level_info = self._get_next_level_model(current_level)
        if next_level_info is None:
            logger.info("Already at maximum level")
            return EscalationDecision(
                should_escalate=False,
                reason=EscalationReason.NO_ESCALATION,
                target_level=current_level,
                target_model_id=current_result.chosen_model_id,
                current_confidence=confidence,
                expected_improvement=0.0,
                expected_additional_cost=0.0,
                evi_score=0.0,
                decision_confidence=confidence
            )
        
        next_level, next_model_id = next_level_info
        
        # Compute EVI for reference
        evi = self._compute_evi(distance, current_level, next_level, next_model_id, query)
        expected_improvement = self._estimate_quality_improvement(
            distance, current_level, next_level
        )
        expected_cost = self.models[next_model_id].estimate_cost(query)
        
        # ====================================================================
        # ESCALATION RULES (in priority order)
        # ====================================================================
        
        # Rule 1: Critical task + low confidence
        if state.is_critical and confidence < self.critical_confidence_threshold:
            logger.warning(
                f"CRITICAL TASK with low confidence ({confidence:.3f}), escalating"
            )
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.CRITICAL_TASK,
                target_level=next_level,
                target_model_id=next_model_id,
                current_confidence=confidence,
                expected_improvement=expected_improvement,
                expected_additional_cost=expected_cost,
                evi_score=evi,
                decision_confidence=0.95
            )
        
        # Rule 2: Intent repetition + low confidence
        if (state.current_repetition_count >= self.intent_repetition_threshold and
            confidence < self.confidence_threshold):
            logger.warning(
                f"INTENT REPETITION ({state.current_repetition_count}x) "
                f"with low confidence ({confidence:.3f}), escalating"
            )
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.INTENT_REPETITION,
                target_level=next_level,
                target_model_id=next_model_id,
                current_confidence=confidence,
                expected_improvement=expected_improvement,
                expected_additional_cost=expected_cost,
                evi_score=evi,
                decision_confidence=0.90
            )
        
        # Rule 3: High prediction error (selector miscalibrated)
        avg_pred_error = state.get_average_prediction_error(window=5)
        if (avg_pred_error > self.high_prediction_error_threshold and
            len(state.prediction_errors) >= 3):
            logger.warning(
                f"HIGH PREDICTION ERROR (avg={avg_pred_error:.3f}), "
                f"selector may be miscalibrated, escalating"
            )
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.HIGH_PREDICTION_ERROR,
                target_level=next_level,
                target_model_id=next_model_id,
                current_confidence=confidence,
                expected_improvement=expected_improvement,
                expected_additional_cost=expected_cost,
                evi_score=evi,
                decision_confidence=0.85
            )
        
        # Rule 4: Positive EVI (rational escalation)
        if evi > 0:
            logger.info(f"POSITIVE EVI ({evi:.4f}), escalating")
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.POSITIVE_EVI,
                target_level=next_level,
                target_model_id=next_model_id,
                current_confidence=confidence,
                expected_improvement=expected_improvement,
                expected_additional_cost=expected_cost,
                evi_score=evi,
                decision_confidence=0.80
            )
        
        # Rule 5: General low confidence
        if confidence < self.confidence_threshold:
            logger.info(f"LOW CONFIDENCE ({confidence:.3f}), escalating")
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                target_level=next_level,
                target_model_id=next_model_id,
                current_confidence=confidence,
                expected_improvement=expected_improvement,
                expected_additional_cost=expected_cost,
                evi_score=evi,
                decision_confidence=0.75
            )
        
        # ====================================================================
        # HYSTERESIS CHECK
        # ====================================================================
        
        # Check if we should maintain current level even with good confidence
        # (prevents rapid de-escalation)
        trend = state.get_confidence_trend(self.hysteresis_window)
        if current_level > 0 and trend == "consistently_high":
            logger.debug(
                f"Confidence consistently high, could consider de-escalation "
                f"in future (not implemented)"
            )
        
        # Default: No escalation needed
        logger.debug(f"No escalation needed (confidence={confidence:.3f})")
        return EscalationDecision(
            should_escalate=False,
            reason=EscalationReason.NO_ESCALATION,
            target_level=current_level,
            target_model_id=current_result.chosen_model_id,
            current_confidence=confidence,
            expected_improvement=0.0,
            expected_additional_cost=0.0,
            evi_score=evi,
            decision_confidence=confidence
        )
    
    def process_query_with_escalation(
        self,
        query: str,
        state: SessionState,
        mode: SelectionMode = SelectionMode.HYBRID,
        force_critical: bool = False,
        max_escalations: int = 2
    ) -> Tuple[SelectionResult, List[EscalationDecision]]:
        """
        Complete query processing with meta-selector oversight.
        
        Algorithm:
        1. Initial selection using selector
        2. Meta-selector validation and escalation decision
        3. If escalate: call higher model, measure actual distance, repeat
        4. Stop when: no escalation needed, max escalations reached, or budget exhausted
        
        Returns:
            (final_result, escalation_history)
        """
        escalation_history = []
        
        # Initial selection
        logger.info(f"Initial selection for query: {query[:50]}...")
        result = self.selector.select(query, mode=mode)
        
        # Escalation loop
        for escalation_round in range(max_escalations):
            logger.info(f"\n--- Escalation Round {escalation_round + 1} ---")
            
            # Meta-selector decision
            decision = self.validate_and_decide(query, result, state, force_critical)
            escalation_history.append(decision)
            
            if not decision.should_escalate:
                logger.info("Meta-selector: No escalation needed")
                break
            
            logger.info(
                f"Meta-selector: ESCALATE from {result.chosen_model_id} "
                f"(L{result.chosen_level}) to {decision.target_model_id} "
                f"(L{decision.target_level}) - Reason: {decision.reason.value}"
            )
            
            # Check budget before escalating
            if state.budget_remaining < decision.expected_additional_cost:
                logger.warning("Insufficient budget for escalation")
                break
            
            # Perform escalation: actually call the higher-level model
            query_embedding = self.space.embed_query(query)
            target_adapter = self.models[decision.target_model_id]
            
            # ACTUALLY CALL the escalated model
            output = target_adapter.infer(query)
            
            # Measure TRUE behavioral distance
            output_embedding = self.space.embed_output(decision.target_model_id, output)
            distance = self.space.distance(query_embedding, output_embedding)
            confidence = self.selector._compute_confidence(distance)
            cost = target_adapter.estimate_cost(query)
            
            # Update prototypes with escalated result (continuous learning)
            self.selector.prototypes.add_observation(
                decision.target_model_id,
                output_embedding,
                output,
                self.space.distance
            )
            
            # Create new result
            result = SelectionResult(
                chosen_model_id=decision.target_model_id,
                chosen_level=decision.target_level,
                selection_mode=SelectionMode.EXPLORE,  # Escalation is exploration
                actual_output=output,
                output_embedding=output_embedding,
                behavioral_distance=distance,
                estimated_cost=cost,
                confidence=confidence,
                alternatives_considered=[(decision.target_model_id, distance, cost)],
                prediction_error=None
            )
            
            # Check if escalation improved things
            improvement = decision.current_confidence - confidence
            logger.info(
                f"Escalation result: confidence={confidence:.3f} "
                f"(improvement={-improvement:.3f})"
            )
            
            if confidence >= self.confidence_threshold:
                logger.info("Confidence threshold reached, stopping escalation")
                break
        
        return result, escalation_history


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    from selector_v2 import SimpleEmbeddingSpace, PrototypeBank, create_mock_model
    
    print("=" * 80)
    print("Meta-Selector v2 - Updated Implementation")
    print("Following Revised Theoretical Framework")
    print("=" * 80)
    
    # Setup
    space = SimpleEmbeddingSpace()
    
    models = {
        "small": ModelAdapter(
            model_id="small",
            inference_fn=create_mock_model("small", 0.3),
            cost_per_1k_tokens=0.0001,
            level=0
        ),
        "medium": ModelAdapter(
            model_id="medium",
            inference_fn=create_mock_model("medium", 0.6),
            cost_per_1k_tokens=0.0005,
            level=1
        ),
        "large": ModelAdapter(
            model_id="large",
            inference_fn=create_mock_model("large", 0.9),
            cost_per_1k_tokens=0.002,
            level=2
        ),
    }
    
    # Build selector with prototypes
    prototype_bank = PrototypeBank()
    selector = BehavioralSelector(
        embedding_space=space,
        model_adapters=models,
        prototype_bank=prototype_bank,
        exploration_rate=0.15
    )
    
    # Build prototypes offline
    print("\n" + "=" * 80)
    print("Offline Phase: Building Prototypes")
    print("=" * 80)
    
    training_corpus = [
        "What is quantum mechanics?",
        "Explain AI briefly.",
        "What is 2+2?",
        "Describe relativity.",
        "How does photosynthesis work?",
    ]
    
    selector.build_prototypes(training_corpus, verbose=False)
    
    # Create meta-selector
    meta_selector = MetaSelector(
        selector=selector,
        embedding_space=space,
        model_adapters=models
    )
    
    # Test scenarios
    print("\n" + "=" * 80)
    print("Online Phase: Processing Queries with Meta-Selector")
    print("=" * 80)
    
    scenarios = [
        ("What is 5+5?", False),  # Simple, should use small
        ("Explain quantum entanglement.", False),  # Complex, may escalate
        ("Explain quantum physics in extreme detail.", True),  # Critical, will escalate
        ("Tell me more about quantum mechanics.", False),  # Repetition test
        ("Go deeper into quantum theory.", False),  # More repetition
    ]
    
    session = SessionState(initial_budget=1.0)
    
    for i, (query, is_critical) in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}: {query}")
        print(f"Critical: {is_critical}")
        print(f"{'='*80}")
        
        result, escalations = meta_selector.process_query_with_escalation(
            query=query,
            state=session,
            mode=SelectionMode.HYBRID,
            force_critical=is_critical,
            max_escalations=2
        )
        
        print(f"\n✓ Final Model: {result.chosen_model_id} (Level {result.chosen_level})")
        print(f"  Distance: {result.behavioral_distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Cost: ${result.estimated_cost:.6f}")
        print(f"\n  Output: {result.actual_output[:100]}...")
        
        print(f"\n  Escalation Decisions: {len(escalations)}")
        for j, esc in enumerate(escalations, 1):
            status = "ESCALATE" if esc.should_escalate else "STAY"
            print(f"    {j}. {status} - {esc.reason.value}")
            if esc.should_escalate:
                print(f"       EVI={esc.evi_score:.4f}, "
                      f"ΔQ={esc.expected_improvement:.4f}, "
                      f"ΔC=${esc.expected_additional_cost:.6f}")
        
        print(f"\n  Session State:")
        print(f"    Budget: ${session.budget_remaining:.6f} / ${session.initial_budget:.6f}")
        print(f"    Spent: ${session.budget_spent:.6f}")
        print(f"    Repetition: {session.current_repetition_count}")
        print(f"    Confidence trend: {session.get_confidence_trend()}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Session Summary")
    print("=" * 80)
    print(f"Total interactions: {session.total_interactions}")
    print(f"Total cost: ${session.budget_spent:.6f}")
    print(f"Budget remaining: ${session.budget_remaining:.6f}")
    print(f"\nModel usage:")
    for model_id in set(session.model_history):
        count = session.model_history.count(model_id)
        pct = 100 * count / session.total_interactions
        print(f"  {model_id}: {count} times ({pct:.1f}%)")
    
    print(f"\nAverage confidence: {np.mean(session.confidence_history):.3f}")
    print(f"Average distance: {np.mean(session.distance_history):.3f}")
    if session.prediction_errors:
        print(f"Average prediction error: {np.mean(session.prediction_errors):.3f}")
