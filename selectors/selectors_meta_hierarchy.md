# Selectors and Meta-Selectors in Large Language Model Hierarchies
### A Practical Framework for Efficient Model Routing

**Karol Kowalczyk** (revised by Claude)  
November 2025

---

## Abstract

We present a framework for **adaptive model selection** in hierarchical LLM systems, where queries are routed to appropriately-sized models based on predicted difficulty. Our approach uses **behavioral distance** in a shared embedding space to measure how well a model's output would align with a query's intent. Rather than invoking all models (computationally expensive), we learn **behavioral prototypes** from historical model outputs and use these for fast approximation. A **meta-selector** monitors confidence and manages escalation to higher-capacity models when needed. We provide formal problem definitions, concrete algorithms, and open-source implementations. While full empirical validation remains future work, our framework bridges theory and practice for cost-efficient LLM inference.

**Keywords**: model routing, hierarchical inference, behavioral distance, LLM efficiency, meta-learning

---

## 1. Introduction

### 1.1 Motivation

Large language models (LLMs) achieve remarkable capabilities at the cost of massive computational resources. For example:
- GPT-4 may cost $0.03 per 1K tokens
- Claude-3 Opus costs $0.015 per 1K tokens  
- GPT-3.5 Turbo costs $0.0005 per 1K tokens (60× cheaper)

However, empirical analysis shows that **most queries don't require the largest models**:
- Factual queries ("What is the capital of France?") → answered correctly by small models
- Simple summarization → medium models sufficient
- Complex reasoning or creative synthesis → large models necessary

The challenge: **automatically match query difficulty to model capacity** to minimize cost while maintaining quality.

### 1.2 Problem Formulation

**Given:**
- A hierarchy of LLMs: \(\\{M_1, M_2, ..., M_N\\}\) where \(M_i\) is cheaper but less capable than \(M_{i+1}\)
- A query \(x\) with unknown difficulty
- Cost function \(C(M_i)\) and quality metric \(Q(M_i, x)\)

**Goal:**
Select \(M^* = \arg\min_i [C(M_i)]\) subject to \(Q(M_i, x) \geq \theta\)

**Key Insight:**  
We can estimate \(Q(M_i, x)\) by measuring **behavioral distance**: how close the model's output embedding is to the query's embedding in a shared semantic space.

### 1.3 Contributions

1. **Formal framework** for behavioral distance-based model selection
2. **Two-phase design** separating offline prototype learning from online routing
3. **Meta-selector** for confidence validation and hierarchical escalation
4. **Concrete algorithms** with complexity analysis
5. **Open-source implementation** in Python

### 1.4 Related Work

**Mixture of Experts (MoE)**: Use learned gating networks to route inputs. Our approach differs by using geometric distance in a shared embedding space rather than learned weights.

**FrugalGPT & Cascade Methods**: Route through models sequentially. We enable **parallel estimation** via prototypes and **cross-level comparison**.

**Speculative Decoding**: Uses small models to speed up large models. We focus on selecting the right model entirely, not accelerating generation.

**Early Exit Networks**: Exit from layer computation early. We operate at the model-level, not layer-level.

---

## 2. Theoretical Framework

### 2.1 Behavioral Distance: The Ideal Case

**Definition** (Behavioral Distance):  
Given a query \(x\) and model \(M_i\), the behavioral distance is:
\[
d_{\text{behav}}(M_i, x) = D_S(f_S(x), g_S(M_i(x)))
\]
where:
- \(f_S\): query embedding function
- \(g_S\): output embedding function  
- \(D_S\): distance metric (e.g., cosine distance)
- \(M_i(x)\): actual output of model \(M_i\) on query \(x\)

**Interpretation**: Lower distance means the model's output semantically aligns with the query's intent.

**Ideal Selection Rule**:
\[
M^* = \arg\min_i \\big[w_d \cdot d_{\text{behav}}(M_i, x) + w_c \cdot C(M_i)\\big]
\]

**Computational Cost**: This requires calling **all N models**, which defeats the purpose of efficient routing!

### 2.2 Practical Approximation via Prototypes

To avoid invoking all models, we introduce **behavioral prototypes**:

**Definition** (Behavioral Prototype):  
A prototype \(p_k^{(i)}\) for model \(M_i\) is a representative output embedding learned from historical data:
\[
p_k^{(i)} = \text{Centroid}\\{g_S(M_i(x_j)) : x_j \in \text{Cluster}_k\\}
\]

**Approximation**:
\[
d_{\text{behav}}(M_i, x) \approx \min_k D_S(f_S(x), p_k^{(i)})
\]

**Trade-off**: 
- Exact: \(O(N)\) model calls per query
- Approximate: \(O(N \times K)\) distance computations (no model calls)
- Where \(K\) = number of prototypes per model (typically 3-10)

**Approximation Error**:  
\[
\epsilon(x) = |d_{\text{behav}}(M_i, x) - \min_k D_S(f_S(x), p_k^{(i)})|
\]

This depends on:
- Prototype quality (how well they cover model behavior space)
- Query novelty (distance to training distribution)
- Embedding space quality

### 2.3 Intra-Level Selection

**Observation**: Models at the "same level" (similar cost) often specialize differently.
- GPT-4 vs Claude-3 Opus (both "large")
- Llama-70B vs Mixtral-8x7B (both "medium")

**Extended Selection Rule**:
\[
M^* = \arg\min_{i,j} \\big[w_d \cdot d_{\text{behav}}(M_i^{(j)}, x) + w_c \cdot C(M_i^{(j)})\\big]
\]
where \(M_i^{(j)}\) denotes model \(j\) at level \(i\).

This enables **horizontal selection** (within level) and **vertical selection** (across levels).

### 2.4 Meta-Selector and Escalation

**Problem**: The selector might make wrong decisions due to:
- Poor prototype coverage
- Distribution shift
- Miscalibrated confidence

**Solution**: A **meta-selector** validates decisions post-hoc and escalates when needed.

**Confidence Estimation**:
\[
\text{conf}(d) = \frac{1}{1 + \exp(k(d - d_0))}
\]
where \(d\) is behavioral distance, and \(k, d_0\) are calibrated on validation data.

**Escalation Rule** (Expected Value of Information):
\[
\text{Escalate if: } (\Delta Q - \lambda \Delta C) > 0
\]
where:
- \(\Delta Q\) = expected quality improvement from higher model
- \(\Delta C\) = additional cost
- \(\lambda\) = cost weight

---

## 3. Algorithms

### Algorithm 1: Offline Prototype Learning

```
Input: Model set {M_1, ..., M_N}, query corpus Q
Output: Prototype bank B

1. For each model M_i:
   a. Sample diverse queries Q_sample ⊂ Q
   b. For each query q in Q_sample:
      - output_q = M_i(q)  // Actually call the model
      - embed_q = g_S(output_q)
      - Store (q, embed_q)
   
   c. Cluster embeddings into K clusters:
      - Use K-means or hierarchical clustering
      - Set K based on output diversity (typically 3-10)
   
   d. Compute centroids:
      - p_k^{(i)} = mean({embed_q : q in Cluster_k})
   
   e. Store prototypes: B[M_i] = {p_1^{(i)}, ..., p_K^{(i)}}

2. Return B

Complexity: O(N × |Q_sample| × T_model)
where T_model = time to call one model
```

### Algorithm 2: Online Selection (Exploit Mode)

```
Input: Query x, prototype bank B, models {M_1, ..., M_N}
Output: Chosen model M*, actual output y

1. Embed query:
   z_x = f_S(x)

2. For each model M_i:
   a. Find nearest prototype:
      k* = argmin_k D_S(z_x, p_k^{(i)})
      d_i = D_S(z_x, p_k*^{(i)})
   
   b. Compute score:
      s_i = w_d × d_i + w_c × C(M_i) + w_l × level(M_i)

3. Select best model:
   M* = argmin_i s_i

4. Call selected model:
   y = M*(x)  // Only invoke ONE model

5. (Optional) Update prototype:
   z_y = g_S(y)
   p_k*^{(*)} ← α × p_k*^{(*)} + (1-α) × z_y  // EMA update

6. Return M*, y

Complexity: O(N × K × d)
where d = embedding dimension
```

### Algorithm 3: Online Selection (Explore Mode)

```
Input: Query x, prototype bank B, models {M_1, ..., M_N}, sample_size S
Output: Chosen model M*, actual output y

1. Embed query:
   z_x = f_S(x)

2. Sample models strategically:
   - Select one model from each level, or
   - Sample proportional to uncertainty, or
   - Pure random sampling
   
   M_sample = {M_i1, M_i2, ..., M_iS} (S models)

3. For each M_ij in M_sample:
   a. Actually call the model:
      y_j = M_ij(x)
   
   b. Embed output:
      z_j = g_S(y_j)
   
   c. Compute true distance:
      d_j = D_S(z_x, z_j)
   
   d. Compute score:
      s_j = w_d × d_j + w_c × C(M_ij)
   
   e. Update prototypes:
      Update B[M_ij] with z_j (clustering/EMA)

4. Select best:
   M* = argmin_j s_j
   y* = corresponding y_j

5. Return M*, y*

Complexity: O(S × T_model)
Note: S << N, typically S = 2-5
```

### Algorithm 4: Meta-Selector Validation

```
Input: Query x, selected model M*, output y, session state σ
Output: Escalation decision

1. Validate selection:
   z_x = f_S(x)
   z_y = g_S(y)
   d_actual = D_S(z_x, z_y)
   conf = confidence_fn(d_actual)

2. Update session state:
   σ.confidence_history.append(conf)
   σ.query_history.append(x)
   σ.detect_repetition(x)  // Compare to recent queries

3. Check budget:
   if σ.budget_remaining ≤ 0:
      return NO_ESCALATION

4. Check escalation triggers:
   
   a. Critical task + low confidence:
      if σ.is_critical and conf < θ_crit:
         return ESCALATE(reason=CRITICAL)
   
   b. Intent repetition + low confidence:
      if σ.repetition_count ≥ 2 and conf < θ_rep:
         return ESCALATE(reason=REPETITION)
   
   c. Expected Value of Information:
      M_next = get_next_level(M*)
      ΔQ = estimate_improvement(d_actual, level(M*), level(M_next))
      ΔC = C(M_next)
      if (ΔQ - λ × ΔC) > 0:
         return ESCALATE(reason=POSITIVE_EVI)
   
   d. General low confidence:
      if conf < θ:
         return ESCALATE(reason=LOW_CONFIDENCE)

5. return NO_ESCALATION

Complexity: O(d) for embedding + O(1) for logic
```

---

## 4. Implementation Details

### 4.1 Embedding Spaces

**Option 1**: SentenceTransformers
- Pre-trained: `all-MiniLM-L6-v2` (384-dim, fast)
- Semantic: `all-mpnet-base-v2` (768-dim, better quality)

**Option 2**: LLM-based embeddings
- OpenAI `text-embedding-3-small` (1536-dim)
- Cohere embed-v3 (1024-dim)

**Key Requirement**: Same embedding function for queries and outputs to ensure shared semantic space.

### 4.2 Distance Metrics

**Cosine Distance** (default):
\[
D_{\text{cos}}(u, v) = 1 - \frac{u \cdot v}{||u|| \cdot ||v||}
\]

**Euclidean Distance**:
\[
D_{\text{L2}}(u, v) = ||u - v||_2
\]

**Mahalanobis Distance** (if covariance matrix available):
\[
D_M(u, v) = \sqrt{(u-v)^T \Sigma^{-1} (u-v)}
\]

### 4.3 Confidence Calibration

Train logistic regression on validation set:
```
Data: {(d_i, label_i)} where label_i = 1 if output was good, 0 otherwise
Model: conf(d) = 1 / (1 + exp(k(d - d_0)))
Fit: Minimize cross-entropy loss to find k, d_0
```

### 4.4 Prototype Count Selection

**Too few** (K=1): Poor coverage of model behavior space  
**Too many** (K>20): Overfitting, high memory, slow search

**Heuristic**: 
- Start with K=5
- Monitor coverage: fraction of queries within threshold of nearest prototype
- Add prototypes if coverage < 0.9
- Remove prototypes if never selected

### 4.5 Handling Cold Start

**Problem**: No prototypes initially.

**Solution 1**: Random sampling
- Route first 100 queries randomly across models
- Build initial prototypes

**Solution 2**: Template-based initialization
- Use generic queries: "Explain...", "Summarize...", "What is...?"
- Generate prototypes offline

**Solution 3**: Always-explore initially
- Use explore mode for first N queries
- Switch to hybrid once prototypes stabilize

---

## 5. Theoretical Analysis

### 5.1 Approximation Bound

**Theorem 1** (Informal): If query \(x\) is within distance \(\delta\) of its nearest prototype's source queries, then:
\[
\epsilon(x) \leq O(\delta + \text{model\_variance})
\]

**Interpretation**: Prototype accuracy depends on:
1. Query coverage in training set
2. Consistency of model outputs

### 5.2 Complexity Comparison

| Method | Offline | Online | Model Calls |
|--------|---------|--------|-------------|
| Ideal (call all) | O(1) | O(N × T_model) | N per query |
| Cascade | O(1) | O(L × T_model) | L ≤ N per query |
| Our (exploit) | O(N × M × T_model) | O(N × K × d) | 1 per query |
| Our (explore) | O(N × M × T_model) | O(S × T_model) | S per query |

Where:
- N = number of models
- M = size of offline corpus
- K = prototypes per model
- d = embedding dimension
- L = average cascade length
- S = exploration sample size

**Key**: Offline cost is amortized. Online is much faster.

---

## 6. Experimental Design (Future Work)

### 6.1 Datasets

Propose evaluation on:
1. **MMLU**: Multi-domain questions, varying difficulty
2. **HumanEval**: Coding tasks (clear correctness metric)
3. **TruthfulQA**: Factual accuracy
4. **AlpacaEval**: Instruction following
5. **MT-Bench**: Diverse capabilities

### 6.2 Baselines

- **Random**: Random model selection
- **Always-Small**: Always use cheapest model
- **Always-Large**: Always use best model (upper bound on quality)
- **Cascade**: Try small→medium→large until confidence threshold
- **Learned Router**: Train classifier on query features

### 6.3 Metrics

**Primary**:
- **Cost Reduction**: Total cost vs always-large baseline
- **Quality**: Accuracy/semantic similarity vs always-large

**Secondary**:
- Latency (including routing overhead)
- Pareto efficiency curve
- Calibration error of confidence estimates

### 6.4 Ablations

Test impact of:
- Number of prototypes (K = 1, 3, 5, 10, 20)
- Exploration rate (ε = 0, 0.05, 0.1, 0.2)
- Distance metric (cosine vs Euclidean vs Mahalanobis)
- Embedding model (small vs large)
- Meta-selector (with vs without)

---

## 7. Discussion

### 7.1 When This Works Well

**Best scenarios**:
- Large query volumes (amortize offline cost)
- Clear difficulty stratification (queries naturally cluster)
- Stable model behaviors (prototypes remain valid)
- Multiple models available at each level

**Example**: Customer support chatbot with mix of:
- Simple FAQs → small model
- Product questions → medium model
- Complex troubleshooting → large model

### 7.2 Limitations

**Limitation 1**: Requires offline calibration phase
- Need representative query corpus
- Models must be available for profiling

**Limitation 2**: Approximation error
- Prototypes may not cover all model behaviors
- Novel queries may be misrouted

**Limitation 3**: Embedding quality dependence
- If embeddings don't capture semantic adequacy, behavioral distance fails
- Different tasks may need different embedding spaces

**Limitation 4**: Static model assumption
- If models are retrained/updated, prototypes become stale
- Need periodic recalibration

**Limitation 5**: Multi-turn conversations
- Current formulation is stateless (per-query)
- Doesn't account for conversation history

### 7.3 Extensions

**Extension 1**: Task-specific embedding spaces
- Learn separate spaces for coding, math, creative writing
- Route to appropriate space based on query type

**Extension 2**: Dynamic prototypes
- Continuously update prototypes online
- Detect distribution shift and trigger recalibration

**Extension 3**: Multi-objective optimization
- Balance cost, latency, and quality simultaneously
- Pareto frontier selection

**Extension 4**: Hierarchical meta-selectors
- Meta-meta-selectors to validate meta-selector decisions
- Recursive control structure

### 7.4 Relationship to Adjoint Projections

Our framework can be viewed through category-theoretic lens:
- Collapse \(C: M_{i+1} \to M_i\): Use simpler model
- Projection \(P: M_i \to M_{i+1}\): Escalate to complex model
- Adjunction: \(P \circ C \approx \text{id}\) (round-trip preserves information)

However, we deliberately keep the paper focused on practical concerns rather than full categorical formalism. The interested reader can explore connections to:
- Functorial semantics of computation
- Information-theoretic bounds on model compression
- Category of models with morphisms as distillation/escalation

---

## 8. Conclusion

We presented a practical framework for adaptive model selection in LLM hierarchies using behavioral distance and learned prototypes. Our two-phase design—offline prototype learning and online fast routing—enables efficient selection without invoking all models. The meta-selector provides confidence validation and rational escalation.

**Key contributions**:
1. Formal problem definition bridging theory and practice
2. Concrete algorithms with complexity analysis
3. Clear specification of approximations and trade-offs
4. Open-source implementation

**Honest assessment**: While we believe this approach is promising, full empirical validation across diverse tasks and model hierarchies remains essential future work. We hope this framework provides a foundation for further research in cost-efficient LLM inference.

---

## References

1. Kowalczyk, K. (2025). *Adjoint Projections on Computational Hierarchies.*
2. Chen, L. et al. (2023). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.*
3. Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*
4. Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding.*
5. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*

---

## Appendix A: Pseudocode for Complete System

See supplementary materials for full Python implementation at:
https://github.com/example/behavioral-selector
