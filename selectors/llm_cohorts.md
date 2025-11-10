# Sleep–Wake Orchestration in Hierarchical LLM Cohorts

## Abstract
This paper introduces a formal architecture for continuous self-optimization in large language model (LLM) ensembles through alternating sleep–wake cycles. Inspired by biological sleep dynamics and grounded in the theory of *adjoint projections on computational hierarchies*, the method organizes a population of models ("cohorts") into rotating states of **wake** (active inference) and **sleep** (fine-tuning on informational gaps). Each model periodically withdraws from production to retrain on the most informative regions of the problem space—those not effectively covered by peers of similar capacity but solvable by higher-level models. This process emulates how biological systems consolidate sensorimotor predictions through subcortical replay. The architecture is formalized in terms of computational projections, behavioral metrics, and bounded compute budgets (1/3 for tuning, 2/3 for active operation). The resulting system self-organizes toward optimal coverage and energy-efficient reasoning, providing a theoretical and practical foundation for self-maintaining model ecosystems.

**Keywords:** LLM hierarchy, meta-learning, continual learning, adjoint projection, sleep, fine-tuning, behavioral metrics, computational consciousness.

---

## 1. Introduction
The performance of large-scale language models depends not only on parameter size but also on **how information is distributed** across hierarchical computational modules. Unlike static architectures, biological cognition exhibits cyclic phases of activity and consolidation—**wakefulness** (real-time inference) and **sleep** (offline reconfiguration). We propose a computational counterpart of this duality within LLM ensembles.

In our approach, models of different capacities form a *cohort* governed by a **selector** (for task routing) and a **meta-selector** (for performance evaluation and escalation). A fixed fraction of the available compute (1/3) is continuously devoted to models in a *sleep phase*, where they fine-tune (distill) on data extracted from the operational ensemble. These models wake with improved specialization, reducing the need for higher-capacity inference. The process realizes a dynamic form of *computational homeostasis* and aligns with the broader theoretical model of **consciousness as collapsed computational time** (Kowalczyk, K., 2025).

---

## 2. Theoretical Background
### 2.1 *Projection Hierarchies and Computational Levels*
Let \(L_0, L_1, L_2, \dots\) denote levels of computational capacity (e.g., parameter scales). Each model \(M_n\) operates on an effective manifold of tasks \(T_n \subset T\). The **selector** \(S\) maps each input \(x \in T\) to the lowest-level model capable of producing a satisfactory output under the cost-quality constraint:
\[
EVI(x) = \mathbb{E}[Q_{n+1}(x) - Q_n(x)] - \lambda (C_{n+1} - C_n) > 0,
\]
where \(Q\) measures response quality and \(C\) represents computational cost.

### 2.2 *Adjoint Duality*
Following *Adjoint Projections on Computational Hierarchies* (Kowalczyk, K., 2025), each transition between levels \(n \to n+1\) can be represented as a pair of adjoint functors:
\[ C_n \dashv P_n : L_{n+1} \leftrightarrows L_n, \]
where \(C_n\) denotes *collapse* (execution, loss of latent degrees of freedom) and \(P_n\) denotes *projection* (learning or reconstruction). The **sleep phase** corresponds to \(P_n\) (projection/updating), while the **wake phase** corresponds to \(C_n\) (collapse/inference).

---

## 3. Cohort Architecture
A cohort \(\mathcal{C}\) is a set of models sharing comparable computational cost \(\kappa(M)\). Each model maintains a finite set of **behavioral prototypes** \(P_{M,k}\) summarizing typical embedding-space responses. A behavioral distance metric \(d_{Beh}(P_i, P_j)\) defines how distinct their outputs are across the embedding space.

- **Selector**: routes input \(x\) by comparing its embedding \(E(x)\) to prototype centroids; chooses the model with minimal expected behavioral distance.
- **Meta-selector**: monitors empirical success, expected value of improvement, and escalation rates to higher tiers.

The *behavioral space* is defined by an embedding function \(E : X \to \mathbb{R}^d\), allowing geometric comparison of tasks, responses, and prototypes.

---

## 4. Sleep–Wake Dynamics
Each model alternates between **wake** and **sleep** according to a rotation schedule constrained by the global compute budget:

- 1/3 of total memory: training pool (sleeping models)
- 2/3 of total memory: inference pool (working models)

In each cycle:
1. The meta-selector identifies *gaps* in the behavioral manifold—regions \(C_i\) where cohort models fail but higher-level models succeed.
2. The model with highest gap misalignment enters sleep, fine-tuning on examples from these cells.
3. Fine-tuning minimizes weighted KL divergence (distillation) from high-level teachers, with additional prototype and diversity regularization.
4. The tuned model re-enters production via *canary rollout*.

This process continuously rebalances knowledge across the cohort, maintaining equilibrium between specialization and coverage.

---

## 5. Gap Metric and Learning Objective
Define a local gap score:
\[
G(z) = D_Q(z) (1 - \text{cover}(z)) \cdot \text{solvable\_up}(z),
\]
where:
- \(D_Q(z)\): density of queries in embedding space.
- \(\text{cover}(z)\): local success rate of cohort peers.
- \(\text{solvable\_up}(z)\): probability that upper-tier models solved this query.

The fine-tuning objective for a sleeping model \(M_s\) is:
\[
\mathcal{L} = \mathbb{E}_{x \sim C_i}\Big[w(x) \cdot \mathrm{KL}(p_{M_s}(\cdot|x)\,\|\,p_{\text{teacher}}(\cdot|x))\Big] + \lambda\|\tilde P - P^{EMA}\|^2 + \mu\,\text{Div}(M_s,\text{cohort}),
\]
with weights \(w(x) = \alpha G(C_i) + \beta EVI(x) + \gamma \text{conf}(x)\).

This aligns the sleeping model toward *informational gaps* while stabilizing existing behaviors.

---

## 6. Resource Allocation Model
Each model class (3B, 8B, 13B, …) operates in **envelopes**:
- One training (sleeping) model per envelope.
- A working pool consuming twice the compute of the sleeper.

If the global budget is \(B_{tot}\), then:
\[
\frac{1}{3}B_{tot} = \sum_{c \in \text{classes}} N_c^{train} m_{train}(c), \quad
\frac{2}{3}B_{tot} = \sum_{c \in \text{classes}} N_c^{work} m_{infer}(c).
\]
This maintains continuous learning within bounded energy and compute constraints.

---

## 7. Biological Analogy and Computational Justification
The mechanism parallels **sleep-dependent learning** in the brain:
- **Cortical–subcortical consolidation:** auxiliary modules refine predictions based on higher-level errors.
- **Replay mechanisms:** slow-wave neural replay stabilizes distributed representations after active periods (cf. Tononi & Cirelli, 2016).

Analogously, the LLM cohort’s smaller models replay high-value examples from production logs, adjusting low-level weights via distillation from high-level models. Energy-intensive learning happens intermittently (sleep), preserving real-time responsiveness.

---

## 8. Relation to Adjoint Projections
In categorical terms, each sleep–wake cycle realizes a *computational adjunction*:
\[
P : \text{Experience} \to \text{Model}, \quad C : \text{Model} \to \text{Response}, \quad P \dashv C.
\]
Sleep performs **projection** (updating latent representations); wake performs **collapse** (producing concrete outputs). The system oscillates between these dual modes, maintaining bounded yet evolving computational coherence—a formal analog of consciousness as the *collapse of computational time* (Kowalczyk, K., 2025).

---

## 9. Simulation Plan and Evaluation Metrics
To evaluate the architecture, we propose synthetic experiments with parameterized cohorts (1B–13B) under controlled traffic.

**Metrics:**
- Escalation rate: fraction of queries escalated to higher tiers.
- Confidence stability: variance of output entropy across cycles.
- Gap coverage: reduction of uncovered mass \(\sum G(C_i)\).
- Cost per query: mean inference compute over time.

**Hypothesis:**
Periodic fine-tuning on informational gaps will yield sublinear growth of escalation cost while maintaining accuracy.

---

## 10. Discussion and Future Work
The proposed system integrates *continuous self-optimization* with formal constraints on computational adjunctions. It transforms static multi-model routing into a dynamic ecosystem that balances performance, energy, and memory.

Future directions include:
1. Extending the theory to non-stationary environments (domain drift).
2. Introducing differentiable meta-selectors that learn the optimal sleep schedule.
3. Formalizing the entropy of behavioral manifolds as a measure of cognitive diversity.
4. Linking this architecture with mixture-of-experts and retrieval-augmented models for empirical validation.
5. Analyzing stability–plasticity balance as a formal constraint in continual learning.

---

## Appendix A: Simplified Pseudocode
```python
for each cohort in COHORTS:
    update_gap_index(cohort)
    sleeper = select_model_to_sleep(cohort)
    if sleeper:
        data = collect_gap_batches(sleeper.target_cells)
        train_with_KD(sleeper, data)
        rollout_model(sleeper)
    update_prototypes_and_metrics(cohort)
```

---

## References
1. Kowalczyk, K. (2025). *Selectors and Meta-Selectors in Large Language Model Hierarchies.*
2. Kowalczyk, K. (2025). *Adjoint Projections on Computational Hierarchies.*
3. Kowalczyk, K. (2025). *Consciousness as Collapsed Computational Time.*
4. Tyszkiewicz, J. (1998–2010). *Computational Categories and Simulation Semantics.*
5. Tononi, G., & Cirelli, C. (2016). *Sleep and the Price of Plasticity: From Synaptic to Systems Neuroscience.* Neuron, 81(1), 12–34.
6. Luo et al. (2023). *Catastrophic Forgetting in Continual Fine-Tuning of LLMs.*
7. Parthasarathy et al. (2024). *The Ultimate Guide to Fine-Tuning LLMs.*
8. RoSTE (2025). *Quantization-Aware Supervised Fine-Tuning for LLMs.*
9. STABLE (2025). *Gated Continual Learning for Large Language Models.*
10. Balancing Fine-Tuning and RAG (2025). *Dynamic Update Strategies for Recommendation LLMs.*

