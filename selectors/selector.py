"""
selector.py - Updated Implementation Based on Revised Theory

This implementation follows the improved theoretical framework with:
1. Clear separation of offline (prototype learning) and online (selection) phases
2. Explicit two-mode operation: explore (call models) vs exploit (use prototypes)
3. Proper behavioral distance measurement using actual model outputs
4. Transparent trade-offs between accuracy and efficiency

Author: Updated by Claude based on revised framework
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Protocols and Types
# ============================================================================

class ModelInferenceProtocol(Protocol):
    """Protocol for actual model inference."""
    def __call__(self, query: str) -> str:
        """Execute model inference and return output."""
        ...


class CostEstimator(Protocol):
    """Protocol for estimating inference cost."""
    def __call__(self, query: str) -> float:
        """Estimate cost of running model on query."""
        ...


# ============================================================================
# Embedding Space - Shared semantic space for queries and outputs
# ============================================================================

class EmbeddingSpace(ABC):
    """
    Abstract base for embedding queries and outputs in a shared latent space.
    This is the foundation of behavioral distance measurement.
    """
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query into the latent space: f_S(x)"""
        pass
    
    @abstractmethod
    def embed_output(self, model_id: str, output: str) -> np.ndarray:
        """Embed a model output into the latent space: g_S(M_i(x))"""
        pass
    
    @abstractmethod
    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute distance between two embeddings: D_S(u, v)"""
        pass


@dataclass
class SimpleEmbeddingSpace(EmbeddingSpace):
    """
    Simple TF-IDF based embedding space for demonstration.
    Production use should employ SentenceTransformers or LLM embeddings.
    """
    vocab: Dict[str, int] = field(default_factory=dict)
    idf: Dict[str, float] = field(default_factory=dict)
    doc_count: int = 0
    dim: int = 512
    normalize: bool = True
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return text.lower().split()
    
    def _compute_tfidf(self, text: str) -> np.ndarray:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        
        # Update vocabulary and IDF
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in self.vocab and len(self.vocab) < self.dim:
                self.vocab[token] = len(self.vocab)
            if token not in self.idf:
                self.idf[token] = 0
            self.idf[token] += 1
        self.doc_count += 1
        
        # Compute TF
        from collections import Counter
        tf = Counter(tokens)
        
        # Create vector
        vec = np.zeros(self.dim, dtype=np.float32)
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf_val = np.log(self.doc_count / (self.idf[token] + 1))
                vec[idx] = count * idf_val
        
        # Normalize
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
        
        return vec
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using TF-IDF."""
        return self._compute_tfidf(query)
    
    def embed_output(self, model_id: str, output: str) -> np.ndarray:
        """
        Embed model output using TF-IDF.
        Could prepend model_id to learn model-specific styles.
        """
        return self._compute_tfidf(output)
    
    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Cosine distance between embeddings."""
        dot = np.dot(u, v)
        norm_u = np.linalg.norm(u) + 1e-12
        norm_v = np.linalg.norm(v) + 1e-12
        return float(1.0 - dot / (norm_u * norm_v))


# ============================================================================
# Model Adapter - Wraps actual models with metadata
# ============================================================================

@dataclass
class ModelAdapter:
    """
    Adapter for a language model in the hierarchy.
    Provides inference, cost estimation, and metadata.
    """
    model_id: str
    inference_fn: ModelInferenceProtocol
    cost_per_1k_tokens: float
    level: int  # Hierarchy level: 0=small, 1=medium, 2=large
    quality_prior: float = 0.5  # Prior belief about quality [0,1]
    
    def infer(self, query: str) -> str:
        """Execute actual model inference."""
        logger.debug(f"Calling {self.model_id} with query: {query[:50]}...")
        return self.inference_fn(query)
    
    def estimate_cost(self, query: str) -> float:
        """
        Estimate cost based on query length.
        Uses rough heuristic: 4 chars ≈ 1 token.
        """
        query_tokens = len(query) / 4
        # Assume response is similar length
        total_tokens = query_tokens * 2
        return (total_tokens / 1000) * self.cost_per_1k_tokens


# ============================================================================
# Behavioral Prototypes - Learned representations of model behaviors
# ============================================================================

@dataclass
class BehavioralPrototype:
    """
    A prototype representing typical model behavior in embedding space.
    
    Prototypes are learned from actual model outputs, not from queries.
    Each prototype represents a cluster of similar outputs.
    """
    embedding: np.ndarray  # Centroid in embedding space
    count: int = 0  # Number of outputs contributing to this prototype
    variance: float = 0.0  # Variance within cluster (for confidence)
    exemplar_output: Optional[str] = None  # Example output from this cluster


@dataclass
class PrototypeBank:
    """
    Stores and manages behavioral prototypes for all models.
    
    Key operations:
    - Learn prototypes from actual model outputs (offline phase)
    - Predict behavioral distance using prototypes (online phase)
    - Update prototypes with new observations (continuous learning)
    """
    max_prototypes_per_model: int = 5
    ema_decay: float = 0.9  # For EMA updates
    cluster_threshold: float = 0.3  # Cosine distance threshold for new clusters
    
    _bank: Dict[str, List[BehavioralPrototype]] = field(default_factory=dict)
    
    def get_prototypes(self, model_id: str) -> List[BehavioralPrototype]:
        """Get all prototypes for a model."""
        return self._bank.get(model_id, [])
    
    def has_prototypes(self, model_id: str) -> bool:
        """Check if model has any prototypes."""
        return model_id in self._bank and len(self._bank[model_id]) > 0
    
    def add_observation(
        self,
        model_id: str,
        output_embedding: np.ndarray,
        output_text: str,
        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> None:
        """
        Add an observed model output to update prototypes.
        
        This is the core learning mechanism - we build prototypes from
        ACTUAL model outputs, not from queries.
        
        Algorithm:
        1. If no prototypes exist, create first one
        2. Find nearest existing prototype
        3. If distance > threshold and room for more, create new prototype
        4. Otherwise, update nearest prototype with EMA
        """
        if model_id not in self._bank:
            self._bank[model_id] = []
        
        prototypes = self._bank[model_id]
        
        # Case 1: First prototype for this model
        if len(prototypes) == 0:
            self._bank[model_id].append(BehavioralPrototype(
                embedding=output_embedding.copy(),
                count=1,
                variance=0.0,
                exemplar_output=output_text
            ))
            logger.info(f"Created first prototype for {model_id}")
            return
        
        # Case 2: Find nearest existing prototype
        distances = [distance_fn(output_embedding, p.embedding) for p in prototypes]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        
        # Case 3: Create new prototype if sufficiently different
        if (nearest_dist > self.cluster_threshold and 
            len(prototypes) < self.max_prototypes_per_model):
            self._bank[model_id].append(BehavioralPrototype(
                embedding=output_embedding.copy(),
                count=1,
                variance=nearest_dist,  # Initial variance estimate
                exemplar_output=output_text
            ))
            logger.info(
                f"Created prototype {len(self._bank[model_id])} for {model_id} "
                f"(dist={nearest_dist:.3f})"
            )
        else:
            # Case 4: Update nearest prototype with EMA
            proto = prototypes[nearest_idx]
            
            # Update embedding with EMA
            proto.embedding = (
                self.ema_decay * proto.embedding +
                (1.0 - self.ema_decay) * output_embedding
            ).astype(np.float32)
            
            # Update variance (running average)
            proto.variance = (
                self.ema_decay * proto.variance +
                (1.0 - self.ema_decay) * nearest_dist
            )
            
            proto.count += 1
            
            # Periodically update exemplar
            if proto.count % 10 == 0:
                proto.exemplar_output = output_text
            
            logger.debug(
                f"Updated prototype {nearest_idx} for {model_id} "
                f"(count={proto.count}, var={proto.variance:.3f})"
            )
    
    def predict_distance(
        self,
        model_id: str,
        query_embedding: np.ndarray,
        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> Tuple[float, float]:
        """
        Predict behavioral distance for a query using prototypes.
        
        Returns: (predicted_distance, confidence)
        
        This is the approximation that makes online selection efficient:
        instead of calling the model, we estimate the distance to its
        likely output using learned prototypes.
        """
        prototypes = self.get_prototypes(model_id)
        
        if not prototypes:
            # No knowledge about this model
            return float('inf'), 0.0
        
        # Find nearest prototype
        distances = [distance_fn(query_embedding, p.embedding) for p in prototypes]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        
        # Confidence based on prototype variance
        # Low variance = high confidence in prediction
        prototype = prototypes[min_idx]
        confidence = 1.0 / (1.0 + prototype.variance) if prototype.count > 5 else 0.5
        
        return min_dist, confidence


# ============================================================================
# Selection Modes and Results
# ============================================================================

class SelectionMode(Enum):
    """
    Selection strategies implementing explore-exploit trade-off.
    
    EXPLOIT: Fast, uses prototypes, may be inaccurate
    EXPLORE: Slow, calls models, builds accurate prototypes
    HYBRID: Balances both with ε-greedy strategy
    """
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    HYBRID = "hybrid"


@dataclass
class SelectionResult:
    """
    Result of a selection decision.
    Contains both the decision and the actual outcome.
    """
    # Decision
    chosen_model_id: str
    chosen_level: int
    selection_mode: SelectionMode
    
    # Actual outcome (after calling model)
    actual_output: str
    output_embedding: np.ndarray
    
    # Quality metrics
    behavioral_distance: float  # True distance: D_S(query_emb, output_emb)
    estimated_cost: float
    confidence: float  # Calibrated confidence score
    
    # Context
    alternatives_considered: List[Tuple[str, float, float]]  # (id, dist, cost)
    prediction_error: Optional[float] = None  # |predicted_dist - actual_dist|


# ============================================================================
# The Behavioral Selector - Main Implementation
# ============================================================================

class BehavioralSelector:
    """
    Adaptive model selector using behavioral distance in a shared embedding space.
    
    THEORY:
    - Ideal: Evaluate all models, choose best based on behavioral distance + cost
    - Practice: Learn prototypes offline, use for fast online prediction
    
    TWO-PHASE DESIGN:
    
    OFFLINE PHASE (via build_prototypes):
    1. Collect representative queries
    2. Call all models on these queries
    3. Embed actual outputs
    4. Cluster to form K prototypes per model
    → One-time cost: O(N × M × T_model)
    
    ONLINE PHASE (via select):
    1. Embed query: O(d)
    2. Compare to prototypes: O(N × K × d)
    3. Select best model
    4. Call only selected model: O(T_model)
    → Per-query cost: O(N × K × d) + 1 model call
    
    MODES:
    - exploit: Always use prototypes (fastest)
    - explore: Sample models to improve prototypes (learn)
    - hybrid: ε-greedy balance (default)
    """
    
    def __init__(
        self,
        embedding_space: EmbeddingSpace,
        model_adapters: Dict[str, ModelAdapter],
        prototype_bank: PrototypeBank,
        # Scoring weights
        w_distance: float = 1.0,
        w_cost: float = 0.8,
        w_level: float = 0.1,  # Prefer lower levels when tied
        # Confidence
        confidence_threshold: float = 0.75,
        confidence_k: float = -8.0,  # Logistic steepness
        confidence_d0: float = 0.35,  # Logistic midpoint
        # Exploration
        exploration_rate: float = 0.1,  # ε for ε-greedy
        exploration_sample_size: int = 3,
    ):
        self.space = embedding_space
        self.models = model_adapters
        self.prototypes = prototype_bank
        
        self.w_distance = w_distance
        self.w_cost = w_cost
        self.w_level = w_level
        
        self.confidence_threshold = confidence_threshold
        self.confidence_k = confidence_k
        self.confidence_d0 = confidence_d0
        
        self.exploration_rate = exploration_rate
        self.exploration_sample_size = exploration_sample_size
        
        # Statistics
        self.total_selections = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        
        # Get available levels
        self.levels = sorted(set(adapter.level for adapter in model_adapters.values()))
    
    def _compute_confidence(self, distance: float) -> float:
        """
        Convert behavioral distance to confidence score.
        
        Uses logistic function:
        conf(d) = 1 / (1 + exp(k × (d - d_0)))
        
        Lower distance → higher confidence
        """
        return float(1.0 / (1.0 + np.exp(
            self.confidence_k * (distance - self.confidence_d0)
        )))
    
    def _compute_score(
        self,
        distance: float,
        cost: float,
        level: int
    ) -> float:
        """
        Compute composite score for model selection.
        
        score = w_d × distance + w_c × cost + w_l × level
        
        Lower is better (minimization problem).
        """
        return (
            self.w_distance * distance +
            self.w_cost * cost +
            self.w_level * level
        )
    
    def build_prototypes(
        self,
        query_corpus: List[str],
        models_to_profile: Optional[List[str]] = None,
        verbose: bool = True
    ) -> None:
        """
        OFFLINE PHASE: Build behavioral prototypes from actual model outputs.
        
        This is the expensive one-time cost that enables fast online selection.
        
        Algorithm:
        1. For each model to profile:
           a. Call model on all queries in corpus
           b. Embed actual outputs
           c. Add observations to prototype bank (clustering happens automatically)
        
        Complexity: O(|models| × |corpus| × T_model)
        
        Args:
            query_corpus: Representative queries spanning expected distribution
            models_to_profile: Specific models to profile (default: all)
            verbose: Log progress
        """
        if models_to_profile is None:
            models_to_profile = list(self.models.keys())
        
        logger.info(
            f"Building prototypes for {len(models_to_profile)} models "
            f"using {len(query_corpus)} queries"
        )
        
        for model_id in models_to_profile:
            if verbose:
                logger.info(f"Profiling {model_id}...")
            
            adapter = self.models[model_id]
            
            for i, query in enumerate(query_corpus):
                # ACTUALLY CALL THE MODEL
                output = adapter.infer(query)
                
                # Embed the ACTUAL OUTPUT
                output_embedding = self.space.embed_output(model_id, output)
                
                # Add observation to prototype bank
                self.prototypes.add_observation(
                    model_id,
                    output_embedding,
                    output,
                    self.space.distance
                )
                
                if verbose and (i + 1) % 10 == 0:
                    logger.info(
                        f"  {model_id}: {i+1}/{len(query_corpus)} queries processed, "
                        f"{len(self.prototypes.get_prototypes(model_id))} prototypes"
                    )
        
        # Summary
        logger.info("Prototype building complete:")
        for model_id in models_to_profile:
            n_protos = len(self.prototypes.get_prototypes(model_id))
            logger.info(f"  {model_id}: {n_protos} prototypes")
    
    def select_exploit(
        self,
        query: str,
        return_top_k: int = 1
    ) -> SelectionResult:
        """
        EXPLOIT MODE: Use prototypes for fast selection.
        
        Algorithm (Online Phase):
        1. Embed query
        2. For each model:
           a. Predict distance using prototypes (NO model call)
           b. Compute score = w_d × dist + w_c × cost + w_l × level
        3. Select best model
        4. Call ONLY the selected model
        5. Update prototypes with actual output (optional continuous learning)
        
        Complexity: O(N × K × d) + 1 model call
        where N=models, K=prototypes/model, d=embedding dim
        """
        # Step 1: Embed query
        query_embedding = self.space.embed_query(query)
        
        # Step 2: Score all models using prototypes
        candidates = []
        for model_id, adapter in self.models.items():
            # Predict distance using prototypes (no model call!)
            pred_dist, pred_conf = self.prototypes.predict_distance(
                model_id,
                query_embedding,
                self.space.distance
            )
            
            # Estimate cost
            cost = adapter.estimate_cost(query)
            
            # Compute score
            score = self._compute_score(pred_dist, cost, adapter.level)
            
            candidates.append((model_id, pred_dist, cost, score, pred_conf))
        
        # Step 3: Select best
        candidates.sort(key=lambda x: x[3])  # Sort by score
        best_model_id, pred_dist, pred_cost, _, pred_conf = candidates[0]
        
        # Step 4: Call ONLY the selected model
        adapter = self.models[best_model_id]
        output = adapter.infer(query)
        
        # Step 5: Measure TRUE behavioral distance
        output_embedding = self.space.embed_output(best_model_id, output)
        actual_distance = self.space.distance(query_embedding, output_embedding)
        actual_confidence = self._compute_confidence(actual_distance)
        
        # Step 6: Update prototypes with actual behavior (continuous learning)
        self.prototypes.add_observation(
            best_model_id,
            output_embedding,
            output,
            self.space.distance
        )
        
        # Statistics
        self.exploitation_count += 1
        self.total_selections += 1
        
        # Compute prediction error
        prediction_error = abs(pred_dist - actual_distance)
        
        return SelectionResult(
            chosen_model_id=best_model_id,
            chosen_level=adapter.level,
            selection_mode=SelectionMode.EXPLOIT,
            actual_output=output,
            output_embedding=output_embedding,
            behavioral_distance=actual_distance,
            estimated_cost=pred_cost,
            confidence=actual_confidence,
            alternatives_considered=[(m, d, c) for m, d, c, _, _ in candidates[:return_top_k]],
            prediction_error=prediction_error
        )
    
    def select_explore(
        self,
        query: str,
        sample_size: Optional[int] = None
    ) -> SelectionResult:
        """
        EXPLORE MODE: Actually call multiple models to learn behaviors.
        
        Algorithm:
        1. Embed query
        2. Sample S models (typically 2-5)
        3. Call ALL sampled models
        4. Embed all ACTUAL outputs
        5. Measure TRUE behavioral distances
        6. Select best
        7. Update prototypes with all observations
        
        Complexity: O(S × T_model)
        This is expensive but necessary for learning accurate prototypes.
        """
        if sample_size is None:
            sample_size = self.exploration_sample_size
        
        # Step 1: Embed query
        query_embedding = self.space.embed_query(query)
        
        # Step 2: Sample models strategically
        # Strategy: Sample one from each level to ensure coverage
        sampled_models = []
        for level in self.levels:
            level_models = [
                m for m, a in self.models.items() if a.level == level
            ]
            if level_models and len(sampled_models) < sample_size:
                # Could use more sophisticated sampling (e.g., Thompson sampling)
                sampled_models.append(np.random.choice(level_models))
        
        # If we need more, sample randomly
        if len(sampled_models) < sample_size:
            remaining = [
                m for m in self.models.keys() if m not in sampled_models
            ]
            n_more = min(sample_size - len(sampled_models), len(remaining))
            sampled_models.extend(np.random.choice(remaining, n_more, replace=False))
        
        # Step 3-5: Call all sampled models and measure TRUE distances
        evaluations = []
        for model_id in sampled_models:
            adapter = self.models[model_id]
            
            # ACTUALLY CALL THE MODEL
            output = adapter.infer(query)
            
            # Embed ACTUAL OUTPUT
            output_embedding = self.space.embed_output(model_id, output)
            
            # Measure TRUE behavioral distance
            distance = self.space.distance(query_embedding, output_embedding)
            
            # Cost and score
            cost = adapter.estimate_cost(query)
            score = self._compute_score(distance, cost, adapter.level)
            
            evaluations.append((
                model_id, output, output_embedding, distance, cost, score
            ))
            
            # Step 7: Update prototypes with ACTUAL observations
            self.prototypes.add_observation(
                model_id,
                output_embedding,
                output,
                self.space.distance
            )
        
        # Step 6: Select best based on ACTUAL measurements
        evaluations.sort(key=lambda x: x[5])  # Sort by score
        best = evaluations[0]
        best_model_id, output, output_emb, distance, cost, _ = best
        
        confidence = self._compute_confidence(distance)
        
        # Statistics
        self.exploration_count += 1
        self.total_selections += 1
        
        return SelectionResult(
            chosen_model_id=best_model_id,
            chosen_level=self.models[best_model_id].level,
            selection_mode=SelectionMode.EXPLORE,
            actual_output=output,
            output_embedding=output_emb,
            behavioral_distance=distance,
            estimated_cost=cost,
            confidence=confidence,
            alternatives_considered=[(m, d, c) for m, _, _, d, c, _ in evaluations],
            prediction_error=None  # No prediction in explore mode
        )
    
    def select(
        self,
        query: str,
        mode: SelectionMode = SelectionMode.HYBRID,
        force_explore: bool = False
    ) -> SelectionResult:
        """
        Main selection interface with explore-exploit trade-off.
        
        Args:
            query: User query
            mode: EXPLOIT (fast), EXPLORE (accurate), or HYBRID (ε-greedy)
            force_explore: Override mode to force exploration
        
        Returns:
            SelectionResult with chosen model and actual output
        """
        if force_explore:
            mode = SelectionMode.EXPLORE
        
        if mode == SelectionMode.EXPLOIT:
            return self.select_exploit(query)
        
        elif mode == SelectionMode.EXPLORE:
            return self.select_explore(query)
        
        elif mode == SelectionMode.HYBRID:
            # ε-greedy: explore with probability ε
            if np.random.random() < self.exploration_rate:
                logger.debug("HYBRID: Exploring")
                return self.select_explore(query)
            else:
                logger.debug("HYBRID: Exploiting")
                result = self.select_exploit(query)
                
                # Adaptive exploration: if confidence low, consider exploring more
                if result.confidence < self.confidence_threshold:
                    logger.warning(
                        f"Low confidence ({result.confidence:.3f}) - "
                        f"consider increasing exploration rate"
                    )
                
                return result
        
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
    
    def get_statistics(self) -> Dict:
        """Return selector statistics and prototype coverage."""
        stats = {
            "total_selections": self.total_selections,
            "explorations": self.exploration_count,
            "exploitations": self.exploitation_count,
            "exploration_rate": (
                self.exploration_count / max(1, self.total_selections)
            ),
            "models": {}
        }
        
        for model_id, adapter in self.models.items():
            protos = self.prototypes.get_prototypes(model_id)
            stats["models"][model_id] = {
                "level": adapter.level,
                "num_prototypes": len(protos),
                "avg_prototype_count": (
                    np.mean([p.count for p in protos]) if protos else 0
                ),
                "avg_prototype_variance": (
                    np.mean([p.variance for p in protos]) if protos else 0
                )
            }
        
        return stats


# ============================================================================
# Mock Models for Demonstration
# ============================================================================

def create_mock_model(model_id: str, quality_level: float) -> ModelInferenceProtocol:
    """
    Create a mock model with quality-dependent behavior.
    
    Unlike static implementations, these actually process the query
    to produce different outputs based on content.
    """
    def inference(query: str) -> str:
        q_lower = query.lower()
        
        if quality_level < 0.4:  # Small model
            if "quantum" in q_lower:
                return "Quantum particles are connected in special ways."
            elif "what" in q_lower or "explain" in q_lower:
                return "This is a complex topic with many aspects."
            else:
                return f"Brief answer about the topic."
        
        elif quality_level < 0.7:  # Medium model
            if "quantum" in q_lower:
                return (
                    "Quantum entanglement occurs when two particles become "
                    "correlated such that measuring one instantly affects the "
                    "other, even across large distances."
                )
            elif "explain" in q_lower:
                return (
                    "Let me break this down: First, the fundamental concepts "
                    "involve several key principles. Second, these interact in "
                    "specific ways. Third, practical implications emerge from "
                    "this interaction."
                )
            else:
                return f"Detailed explanation addressing the key points raised."
        
        else:  # Large model
            if "quantum" in q_lower:
                return (
                    "Quantum entanglement represents a fundamental departure "
                    "from classical physics, where two particles share a quantum "
                    "state such that measurements on one instantaneously determine "
                    "properties of the other, regardless of separation. This "
                    "non-local correlation, validated by violations of Bell's "
                    "inequalities, doesn't permit faster-than-light signaling but "
                    "reveals the deeply interconnected nature of quantum reality."
                )
            elif "explain" in q_lower:
                return (
                    "To provide a comprehensive explanation, we must consider "
                    "multiple dimensions: (1) Historical context and theoretical "
                    "foundations that established current understanding. (2) Core "
                    "mechanisms and principles that govern the phenomenon. "
                    "(3) Empirical evidence supporting these theories, including "
                    "experimental validations. (4) Practical applications and their "
                    "implications. (5) Current limitations and ongoing research "
                    "directions. Each aspect contributes to a holistic understanding."
                )
            else:
                return (
                    "A thorough analysis requires examining this from multiple "
                    "perspectives, considering both theoretical frameworks and "
                    "empirical observations, while acknowledging the complexity "
                    "and nuance inherent in the subject matter."
                )
    
    return inference


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Behavioral Selector v2 - Updated Implementation")
    print("Following Revised Theoretical Framework")
    print("=" * 80)
    
    # Setup
    space = SimpleEmbeddingSpace()
    
    models = {
        "gpt-small": ModelAdapter(
            model_id="gpt-small",
            inference_fn=create_mock_model("gpt-small", 0.3),
            cost_per_1k_tokens=0.0001,
            level=0,
            quality_prior=0.3
        ),
        "gpt-medium": ModelAdapter(
            model_id="gpt-medium",
            inference_fn=create_mock_model("gpt-medium", 0.6),
            cost_per_1k_tokens=0.0005,
            level=1,
            quality_prior=0.6
        ),
        "gpt-large": ModelAdapter(
            model_id="gpt-large",
            inference_fn=create_mock_model("gpt-large", 0.9),
            cost_per_1k_tokens=0.002,
            level=2,
            quality_prior=0.9
        ),
    }
    
    prototype_bank = PrototypeBank(max_prototypes_per_model=5)
    
    selector = BehavioralSelector(
        embedding_space=space,
        model_adapters=models,
        prototype_bank=prototype_bank,
        exploration_rate=0.2
    )
    
    # ========================================================================
    # OFFLINE PHASE: Build prototypes
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("OFFLINE PHASE: Building Behavioral Prototypes")
    print("=" * 80)
    
    training_corpus = [
        "What is quantum entanglement?",
        "Explain machine learning to a beginner.",
        "What is the capital of France?",
        "How do neural networks work?",
        "Describe the theory of relativity.",
        "What are the benefits of exercise?",
        "Explain photosynthesis.",
        "What is climate change?",
        "How does the internet work?",
        "What is artificial intelligence?",
    ]
    
    print(f"\nTraining corpus: {len(training_corpus)} queries")
    print("This will call ALL models on ALL queries to learn prototypes...\n")
    
    selector.build_prototypes(training_corpus, verbose=True)
    
    # ========================================================================
    # ONLINE PHASE: Fast selection using prototypes
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ONLINE PHASE: Fast Selection Using Learned Prototypes")
    print("=" * 80)
    
    test_queries = [
        "Explain quantum mechanics simply.",
        "What is 2+2?",
        "Describe quantum entanglement in detail.",
        "Give me a brief overview of physics.",
        "Explain the universe.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*80}")
        print(f"Query {i}: {query}")
        print(f"{'-'*80}")
        
        # Force exploration on first query to show both modes
        force_explore = (i == 1)
        
        result = selector.select(
            query,
            mode=SelectionMode.HYBRID,
            force_explore=force_explore
        )
        
        print(f"\n✓ Mode: {result.selection_mode.value.upper()}")
        print(f"  Selected: {result.chosen_model_id} (Level {result.chosen_level})")
        print(f"  Behavioral Distance: {result.behavioral_distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Estimated Cost: ${result.estimated_cost:.6f}")
        
        if result.prediction_error is not None:
            print(f"  Prediction Error: {result.prediction_error:.4f}")
        
        print(f"\n  Output: {result.actual_output[:120]}...")
        
        if len(result.alternatives_considered) > 1:
            print(f"\n  Alternatives:")
            for model_id, dist, cost in result.alternatives_considered[:3]:
                print(f"    {model_id}: dist={dist:.4f}, cost=${cost:.6f}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    
    stats = selector.get_statistics()
    print(f"\nTotal selections: {stats['total_selections']}")
    print(f"Explorations: {stats['explorations']}")
    print(f"Exploitations: {stats['exploitations']}")
    print(f"Exploration rate: {stats['exploration_rate']:.2%}")
    
    print("\nPrototypes per model:")
    for model_id, info in stats['models'].items():
        print(f"  {model_id} (Level {info['level']}):")
        print(f"    Prototypes: {info['num_prototypes']}")
        print(f"    Avg observations per prototype: {info['avg_prototype_count']:.1f}")
        print(f"    Avg prototype variance: {info['avg_prototype_variance']:.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)
