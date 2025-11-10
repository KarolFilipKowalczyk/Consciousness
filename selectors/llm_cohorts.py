"""
llm_cohorts.py

Single-file pseudocode for sleep→tune→wake orchestration of LLM cohorts.
- English-only names and comments
- Combines single-envelope and multi-envelope variants
- Focuses on theoretical planning (not runnable as-is)

Core ideas:
- 1/3 VRAM budget for training (sleep), 2/3 for inference (work)
- A "cohort" is a set of models of the same size class (e.g., 3B, 8B, 13B)
- An "envelope" reserves resources: exactly one sleeper (training) + its workers
- Sleeper fine-tunes adapters on "gap cells" (areas where small models fail but upper level succeeds)
- Canary → rollout → working with safety (hysteresis/stop-loss)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import time
import math
import random


# =========================
# 0) CONFIG / CONSTANTS
# =========================

DEFAULT_SLOT_MINUTES = {
    "3B": 30, "8B": 60, "13B": 90, "30B": 150, "70B": 360, "175B": 720, "GPT5": 1080
}

DEFAULT_LORA_RANK = {
    "3B": 16, "8B": 16, "13B": 16, "30B": 16, "70B": 16, "175B": 16, "GPT5": 16
}

EMA_PROTO = 0.99
EMA_PROTO_GLOBAL = 0.995
KD_TEMPERATURE = 1.7


# =========================
# 1) DATA STRUCTURES
# =========================

class ModelState(str, Enum):
    WORKING = "working"   # handles production traffic
    SLEEPING = "sleeping" # in fine-tuning slot
    ROLLOUT = "rollout"   # canary/expansion after training


@dataclass
class Prototypes:
    # K behavioral prototypes maintained via EMA
    vectors: List[Tuple[str, List[float]]] = field(default_factory=list)  # (id, embedding)

    def update(self, samples: List[List[float]], ema: float = EMA_PROTO_GLOBAL) -> None:
        """Update prototypes with high-confidence samples (EMA)."""
        # Pseudocode: keep simple moving towards new sample means
        # In practice, maintain per-prototype assign/update
        pass


@dataclass
class Model:
    id: str
    size_class: str                 # '3B', '8B', '13B', '30B', '70B', '175B', 'GPT5'
    state: ModelState = ModelState.WORKING
    vr_infer_gb: float = 0.0
    vr_train_gb: float = 0.0
    prototypes: Prototypes = field(default_factory=Prototypes)
    adapters: Dict[str, object] = field(default_factory=dict)  # cell_id -> adapter handle
    metrics: Dict[str, float] = field(default_factory=dict)    # quality/cost metrics
    target_cells: List["Cell"] = field(default_factory=list)
    traffic_share: float = 0.0  # relevant in rollout
    last_sleep_ts: float = 0.0


@dataclass
class Cell:
    id: str
    centroid: List[float]
    gap_weight: float  # G(Ci): demand * (1 - cohort cover) * solvable-up


@dataclass
class GapIndex:
    cells: List[Cell] = field(default_factory=list)

    def update(self, logs_window: list) -> None:
        """Refresh gap cells from recent logs; compute gap weights."""
        # Build/refresh clustering and weights; keep top-k for efficiency
        pass

    def top_cells(self, M: int) -> List[Cell]:
        """Return top-M gap cells by gap_weight."""
        return sorted(self.cells, key=lambda c: c.gap_weight, reverse=True)[:M]

    def weight(self, cell_id: str) -> float:
        for c in self.cells:
            if c.id == cell_id:
                return c.gap_weight
        return 0.0


@dataclass
class Selector:
    """Behavioral selector using prototypes for routing decisions."""
    def route(self, x) -> Model:
        raise NotImplementedError


@dataclass
class MetaSelector:
    """Meta-selector tracking EVI (expected value of improvement) and escalations."""
    def evi_gain(self, x) -> float:
        return 1.0  # placeholder


@dataclass
class Cohort:
    size_class: str
    models: List[Model] = field(default_factory=list)
    gap_index: GapIndex = field(default_factory=GapIndex)
    router: Selector = field(default_factory=Selector)
    meta: MetaSelector = field(default_factory=MetaSelector)


@dataclass
class ResourcePools:
    # Global pools: ~1/3 training VRAM, ~2/3 inference VRAM
    train_free: float
    infer_free: float


@dataclass
class Envelope:
    id: str
    cohort: Cohort
    sleeper: Optional[Model] = None
    workers: List[Model] = field(default_factory=list)
    train_budget_gb: float = 0.0          # = m_train[cohort.size_class]
    infer_budget_gb: float = 0.0          # = 2 * train_budget_gb
    state: str = "idle"                   # 'idle' | 'sleeping' | 'rollout' | 'working'
    target_cells: List[Cell] = field(default_factory=list)
    slot_ends_at: float = 0.0


# =========================
# 2) HELPERS (ABSTRACTED OPS)
# =========================

def collect_logs(cohort: Cohort, window_hours: int = 4) -> list:
    """Pull a sliding window of routing/quality logs."""
    return []

def now_ts() -> float:
    return time.time()

def normalize(x: float) -> float:
    return 0.0 if math.isinf(x) or math.isnan(x) else max(0.0, min(1.0, x))

def success_prob(model: Model, cell: Cell, logs: list) -> float:
    """P(success | cell, model) estimated from logs."""
    return 0.0

def cohort_success_best(cohort: Cohort, cell: Cell, logs: list) -> float:
    """Best success across cohort for a cell."""
    return 0.0

def high_level_teacher(x) -> Tuple[object, float]:
    """Return (teacher_output, teacher_confidence) from upper-level ensemble."""
    return object(), 0.9

def conf_weight(conf: float) -> float:
    return max(0.0, conf)

def evi_gain(x) -> float:
    return 1.0

def sample_from_cell(cell: Cell, n_min: int = 100) -> list:
    """Return at least n_min recent examples from this cell."""
    return []

def steps_for_slot(slot_minutes: int) -> int:
    """Translate a slot duration into a nominal number of optimizer steps."""
    return int(slot_minutes * 60)  # placeholder: 1 step/sec

def pick_lr(size_class: str) -> float:
    return 1e-4

def rank_for_class(cohort: Cohort) -> int:
    return DEFAULT_LORA_RANK.get(cohort.size_class, 16)

def update_prototypes(model: Model, cell: Cell, ema: float = EMA_PROTO) -> None:
    """Update model prototypes with cell-specific targets."""
    pass

def evaluate_on_cells(model: Model, cells: List[Cell], horizon: str = "short") -> dict:
    """Return stats used to decide rollout progression."""
    return {"improves": True, "regress_outside": False}

def improves(stats: dict) -> bool:
    return bool(stats.get("improves", False))

def no_regress_outside(model: Model) -> bool:
    return True

def increase_traffic(model: Model, factor: float, cap: float) -> None:
    model.traffic_share = min(cap, model.traffic_share * factor if model.traffic_share else 0.02)

def stable_for_slots(model: Model, k: int = 2) -> bool:
    return True

def start_canary(model: Model, cells: List[Cell], traffic_share: float = 0.02) -> None:
    model.traffic_share = traffic_share

def rollback_adapters(model: Model) -> None:
    """Disable most-recent adapters from the last sleep slot."""
    pass

def publish_metrics(cohort: Cohort, metrics: Dict[str, float]) -> None:
    pass

def current_escalations(cohort: Cohort) -> float:
    return 0.0

def avg_confidence(cohort: Cohort) -> float:
    return 0.0

def gap_coverage_gain(cohort: Cohort) -> float:
    return 0.0

def cost_per_query(cohort: Cohort) -> float:
    return 0.0

def quality_cost_ratio(model: Model) -> float:
    return model.metrics.get("quality", 1.0) / max(1e-6, model.vr_infer_gb)

def slot_minutes(cohort: Cohort) -> int:
    return DEFAULT_SLOT_MINUTES.get(cohort.size_class, 60)


# Loss components (placeholders)
def KD(pred, tgt, T: float, weight: float) -> float:
    return weight  # placeholder

def PROTO_REG(model: Model, ema: float = EMA_PROTO) -> float:
    return 0.0

def DIVERSITY(cohort: Cohort, pred_logits, cell: Cell) -> float:
    return 0.0


# =========================
# 3) SINGLE-ENVELOPE CORE
# =========================

def gap_misalignment_score(model: Model, cohort: Cohort, logs: list) -> float:
    """How poorly model covers gap cells vs. cohort best; weight by gap mass."""
    score = 0.0
    for cell in cohort.gap_index.top_cells(M=20):
        cover = success_prob(model, cell, logs)
        best = cohort_success_best(cohort, cell, logs)
        score += cell.gap_weight * max(0.0, best - cover)
    return score

def can_allocate_train(model: Model, train_budget_gb: float) -> bool:
    return model.vr_train_gb <= train_budget_gb

def rotate_sleepers(cohort: Cohort) -> None:
    """Pick one sleeper in the cohort (if none sleeping) with highest priority."""
    if any(m.state == ModelState.SLEEPING for m in cohort.models):
        return
    logs = collect_logs(cohort, window_hours=4)
    cohort.gap_index.update(logs)

    candidates: List[Tuple[float, Model]] = []
    for m in cohort.models:
        if m.state != ModelState.WORKING:
            continue
        H_m = gap_misalignment_score(m, cohort, logs)
        recency = now_ts() - m.last_sleep_ts
        score = 0.7 * H_m + 0.3 * normalize(recency)
        candidates.append((score, m))
    if not candidates:
        return
    candidates.sort(key=lambda t: t[0], reverse=True)
    sleeper = candidates[0][1]

    # Assume a per-cohort training envelope (single-envelope variant)
    train_budget_gb = sleeper.vr_train_gb
    if not can_allocate_train(sleeper, train_budget_gb):
        return

    sleeper.state = ModelState.SLEEPING
    sleeper.last_sleep_ts = now_ts()
    sleeper.target_cells = cohort.gap_index.top_cells(M=5)

def run_sleep_training(cohort: Cohort) -> None:
    sleeper = next((m for m in cohort.models if m.state == ModelState.SLEEPING), None)
    if not sleeper:
        return

    data: Dict[str, list] = {}
    for cell in sleeper.target_cells:
        examples = sample_from_cell(cell, n_min=100)
        batches = []
        for x in examples:
            teacher_y, teacher_conf = high_level_teacher(x)
            w = cell.gap_weight * evi_gain(x) * conf_weight(teacher_conf)
            batches.append((x, teacher_y, w))
        data[cell.id] = batches

    for cell in sleeper.target_cells:
        adapter = sleeper.adapters.get(cell.id) or object()  # placeholder
        # Train adapter (pseudocode)
        for _ in range(steps_for_slot(slot_minutes(cohort))):
            # forward -> pred_logits (omitted)
            # loss = KD + PROTO_REG + DIVERSITY
            pass
        update_prototypes(sleeper, cell, ema=EMA_PROTO)

def slot_done(model: Model, cohort: Cohort) -> bool:
    # Placeholder: assume slot length controls when training ends externally
    return True

def manage_rollouts(cohort: Cohort) -> None:
    sleeper = next((m for m in cohort.models if m.state == ModelState.SLEEPING and slot_done(m, cohort)), None)
    if sleeper:
        sleeper.state = ModelState.ROLLOUT
        start_canary(sleeper, cells=sleeper.target_cells, traffic_share=0.02)

    for m in (x for x in cohort.models if x.state == ModelState.ROLLOUT):
        stats = evaluate_on_cells(m, m.target_cells, horizon="short")
        if improves(stats) and no_regress_outside(m):
            increase_traffic(m, factor=2.0, cap=0.5)
            if stable_for_slots(m, k=2):
                m.state = ModelState.WORKING
        else:
            rollback_adapters(m)
            m.state = ModelState.WORKING

def refresh_metrics_and_prototypes(cohort: Cohort) -> None:
    high_conf_samples = []  # gather outside this stub
    for m in cohort.models:
        m.prototypes.update(high_conf_samples, ema=EMA_PROTO_GLOBAL)
    publish_metrics(cohort, {
        "escalation_rate": current_escalations(cohort),
        "avg_confidence":  avg_confidence(cohort),
        "gap_coverage":    gap_coverage_gain(cohort),
        "cost_per_query":  cost_per_query(cohort),
    })

def single_envelope_tick(cohort: Cohort) -> None:
    rotate_sleepers(cohort)
    run_sleep_training(cohort)
    manage_rollouts(cohort)
    refresh_metrics_and_prototypes(cohort)


# =========================
# 4) MULTI-ENVELOPE ORCHESTRATION
# =========================

def gap_pressure(cohort: Cohort) -> float:
    """Priority score for allocating envelopes to a cohort."""
    # Sum of top gap weights; could include request rate, EVI/cost scaling
    return sum(c.gap_weight for c in cohort.gap_index.top_cells(M=10))

def avg_train_mem(cohort: Cohort) -> float:
    # Approximate by average sleeper footprint in this cohort
    vals = [m.vr_train_gb for m in cohort.models]
    return sum(vals) / max(1, len(vals))

def avg_infer_mem(cohort: Cohort) -> float:
    vals = [m.vr_infer_gb for m in cohort.models]
    return sum(vals) / max(1, len(vals))

def ensure_envelopes(cohorts: List[Cohort], pools: ResourcePools) -> List[Envelope]:
    envelopes: List[Envelope] = []
    ranked = sorted(cohorts, key=gap_pressure, reverse=True)
    for c in ranked:
        mtr = avg_train_mem(c)
        minf = avg_infer_mem(c)
        # Max envelopes this cohort can get with remaining pools
        emax = min(int(pools.train_free // mtr), int(pools.infer_free // (2 * minf))) if mtr > 0 and minf > 0 else 0
        for _ in range(max(0, emax)):
            env = Envelope(
                id=f"env-{c.size_class}-{len(envelopes)+1}",
                cohort=c,
                train_budget_gb=mtr,
                infer_budget_gb=2 * mtr,
                state="idle"
            )
            envelopes.append(env)
            pools.train_free -= mtr
            pools.infer_free -= 2 * minf  # conservative
    return envelopes

def pick_sleeper(cohort: Cohort) -> Optional[Model]:
    logs = collect_logs(cohort, window_hours=4)
    cohort.gap_index.update(logs)
    candidates: List[Tuple[float, Model]] = []
    for m in cohort.models:
        if m.state != ModelState.WORKING:
            continue
        H_m = gap_misalignment_score(m, cohort, logs)
        recency = now_ts() - m.last_sleep_ts
        score = 0.7 * H_m + 0.3 * normalize(recency)
        candidates.append((score, m))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

def fits_train_pool(model: Model, train_budget_gb: float) -> bool:
    return model.vr_train_gb <= train_budget_gb

def pick_workers_for_envelope(cohort: Cohort, exclude: List[Model], infer_budget_gb: float) -> List[Model]:
    candidates = [m for m in cohort.models if m.state == ModelState.WORKING and m not in exclude]
    candidates.sort(key=lambda m: quality_cost_ratio(m), reverse=True)
    workers: List[Model] = []
    used = 0.0
    for m in candidates:
        if used + m.vr_infer_gb <= infer_budget_gb:
            workers.append(m)
            used += m.vr_infer_gb
        if used >= infer_budget_gb:
            break
    return workers

def mark_training_alloc(env: Envelope, sleeper: Model) -> None:
    # Attach training resources to sleeper for the slot
    pass

def release_training_alloc(env: Envelope) -> None:
    pass

def run_sleep_training_for_envelope(env: Envelope) -> None:
    m = env.sleeper
    if not m:
        return
    c = env.cohort
    data: Dict[str, list] = {}
    for cell in env.target_cells:
        ex = sample_from_cell(cell, n_min=100)
        batches = []
        for x in ex:
            y_star, conf = high_level_teacher(x)
            w = cell.gap_weight * evi_gain(x) * conf_weight(conf)
            batches.append((x, y_star, w))
        data[cell.id] = batches

    for cell in env.target_cells:
        adapter = m.adapters.get(cell.id) or object()  # placeholder
        for _ in range(steps_for_slot(slot_minutes(c))):
            # forward/backward with KD + PROTO_REG + DIVERSITY
            pass
        update_prototypes(m, cell, ema=EMA_PROTO)

def start_envelope_canary(env: Envelope) -> None:
    if not env.sleeper:
        return
    start_canary(env.sleeper, cells=env.target_cells, traffic_share=0.02)
    env.state = "rollout"

def promote_to_worker(env: Envelope) -> None:
    if not env.sleeper:
        return
    env.sleeper.state = ModelState.WORKING
    env.sleeper.traffic_share = 0.0
    env.sleeper = None
    env.state = "working"

def envelope_tick(env: Envelope, pools: ResourcePools) -> None:
    c = env.cohort
    now = now_ts()

    if env.state in ("idle", "working"):
        sleeper = pick_sleeper(c)
        if sleeper and fits_train_pool(sleeper, env.train_budget_gb):
            env.sleeper = sleeper
            env.sleeper.state = ModelState.SLEEPING
            env.sleeper.last_sleep_ts = now
            env.target_cells = c.gap_index.top_cells(M=5)
            env.slot_ends_at = now + 60 * slot_minutes(c)
            env.state = "sleeping"
            mark_training_alloc(env, env.sleeper)

            env.workers = pick_workers_for_envelope(
                cohort=c,
                exclude=[env.sleeper],
                infer_budget_gb=env.infer_budget_gb
            )
            # route non-target traffic to env.workers (omitted)

    elif env.state == "sleeping":
        run_sleep_training_for_envelope(env)
        if now >= env.slot_ends_at:
            start_envelope_canary(env)

    elif env.state == "rollout":
        if not env.sleeper:
            env.state = "working"
            return
        stats = evaluate_on_cells(env.sleeper, env.target_cells, horizon="short")
        if improves(stats) and no_regress_outside(env.sleeper):
            increase_traffic(env.sleeper, factor=2.0, cap=0.5)
            if stable_for_slots(env.sleeper, k=2):
                promote_to_worker(env)
                release_training_alloc(env)
        else:
            rollback_adapters(env.sleeper)
            env.sleeper.state = ModelState.WORKING
            env.state = "working"
            release_training_alloc(env)

def global_tick(cohorts: List[Cohort], pools: ResourcePools) -> None:
    # 0) refresh gaps per cohort
    for cohort in cohorts:
        logs = collect_logs(cohort, window_hours=4)
        cohort.gap_index.update(logs)

    # 1) allocate/refresh envelopes from pools
    envelopes = ensure_envelopes(cohorts, pools)

    # 2) tick each envelope independently
    for env in envelopes:
        envelope_tick(env, pools)

    # 3) global prototype refresh + metrics
    for cohort in cohorts:
        refresh_metrics_and_prototypes(cohort)
