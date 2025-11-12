# Quick Reference: Entropic Scaling Formulas

## Standard Notation Guide

Use these formulas consistently across all documents:

---

## 1. Information Capacity

**Formula:**
```latex
I(n) = \kappa n \log n \text{ bits}
```

**Usage:**
- Machine $M_n$ has effective information capacity $I(n) = \kappa n \log n$ bits
- For $\kappa = 1$ and log base 2

**Examples:**
- $M_{10}$: $I(10) = 10\kappa \log 10 \approx 33\kappa$ bits
- $M_{20}$: $I(20) = 20\kappa \log 20 \approx 86\kappa$ bits
- $M_{30}$: $I(30) = 30\kappa \log 30 \approx 147\kappa$ bits

---

## 2. Temporal Scaling

**Formula:**
```latex
\tau(n) = \tau_0 + \gamma n \log n \text{ ms}
```

**Usage:**
- Processing time scales entropically
- Temporal integration windows follow $\tau(n)$
- $\tau_0$ is base time, $\gamma$ is scaling factor

---

## 3. Cost Functions

**Formula:**
```latex
C(n) = C_0(1 + \beta n \log n)
```

**Usage:**
- Computational cost grows entropically
- Resource allocation follows this pattern
- $\beta$ controls scaling rate

---

## 4. Python Implementation

**Code:**
```python
import math

def information_capacity(n, kappa=1):
    """Calculate entropic information capacity."""
    if n <= 0:
        return 1
    return int(kappa * n * math.log2(n + 1))

def processing_time(n, tau_0=100, gamma=10):
    """Calculate entropic processing time in ms."""
    if n <= 0:
        return tau_0
    return tau_0 + gamma * n * math.log2(n + 1)
```

---

## 5. LaTeX Table Values

For tables showing capacity levels:

```latex
\begin{tabular}{lll}
Level & Capacity & Example Function \\
\hline
Low (n≈10-15) & ~33-66κ bits & Sensory processing \\
Medium (n≈20-25) & ~86-115κ bits & Working memory \\
High (n≈30-35) & ~147-180κ bits & Abstract reasoning \\
\end{tabular}
```

---

## 6. Common Replacements

### Find and Replace Patterns:

**Capacity:**
- `$2^n$ bits` → `$\kappa n \log n$ bits`
- `$M_n$ has $2^n$ bits` → `$M_n$ has effective information capacity $I(n) = \kappa n \log n$ bits`

**Scaling:**
- `exponentially growing` → `entropically growing`
- `exponential scaling` → `entropic scaling`
- `exponentially increasing` → `entropically increasing`

**Temporal:**
- `$n^2$ time` → `$\tau_0 + \gamma n \log n$ time`
- `quadratic complexity` → `entropic complexity $O(n \log n)$`

---

## 7. When NOT to Replace

**Keep exponential references when:**
1. Comparing to IIT (which uses $2^n$ for Φ calculation)
2. Describing other theories' approaches
3. Showing what we're NOT doing
4. Historical context about exponential growth problems

**Example (CORRECT):**
```latex
Unlike IIT which requires exponential scaling ($2^n$), our framework 
uses entropic scaling ($n \log n$), making it biologically plausible.
```

---

## 8. Verification Checklist

Before finalizing any document:

- [ ] All capacity formulas use $I(n) = \kappa n \log n$
- [ ] All temporal formulas use $\tau(n) = \tau_0 + \gamma n \log n$
- [ ] No $2^n$ except in comparisons to other theories
- [ ] Examples use realistic values (33κ, 86κ, 147κ)
- [ ] Code uses `kappa * n * math.log2(n + 1)`
- [ ] "exponential" only used for contrast/comparison
- [ ] All tables updated with entropic values
- [ ] Cross-references remain consistent

---

## 9. Numerical Quick Reference

### Entropic vs Exponential (κ=1, log₂):

| n | Exponential (2ⁿ) | Entropic (n log n) | Ratio |
|---|------------------|-------------------|-------|
| 5 | 32 | ~12 | 2.7× |
| 10 | 1,024 | ~33 | 31× |
| 15 | 32,768 | ~59 | 555× |
| 20 | 1,048,576 | ~86 | 12,193× |
| 25 | 33,554,432 | ~116 | 289,262× |
| 30 | 1,073,741,824 | ~147 | 7,304,707× |

**Key insight:** By n=30, entropic is ~7 MILLION times smaller!

---

## 10. Common Contexts

### Algorithm pseudocode:
```latex
\State $\text{memory} \gets \text{allocate}(\kappa n \log n \text{ bits})$
```

### Hypothesis statements:
```latex
\textbf{Hypothesis}: Performance shows discrete jumps at capacity 
boundaries corresponding to entropic information levels 
$I(n) = \kappa n \log n$.
```

### Resource descriptions:
```latex
The brain implements a discrete hierarchy with entropically growing 
resources, where $M_n$ has effective information capacity 
$I(n) = \kappa n \log n$ bits.
```

### Predictions:
```latex
Temporal integration windows follow entropic scaling pattern 
$\tau(n) = \tau_0 + \gamma n \log n$ ms.
```

---

## Mathematical Properties

### Growth Rate:
- **Exponential:** $O(2^n)$ - superexponential
- **Entropic:** $O(n \log n)$ - superlinear, subquadratic
- **Linear:** $O(n)$ - baseline

### Derivatives:
```latex
\frac{d}{dn}(n \log n) = \log n + 1
```

This means marginal cost increases logarithmically, much slower than exponential.

---

## Symbol Guide

- **κ** (kappa): Information scaling parameter
- **γ** (gamma): Temporal scaling parameter  
- **β** (beta): Cost scaling parameter
- **τ** (tau): Time duration
- **I(n)**: Information capacity function
- **M_n**: Machine at level n

---

## Citation Style

When citing the entropic scaling:

```latex
We use entropic information scaling $I(n) = \kappa n \log n$, which 
provides super-linear growth without exponential explosion, ensuring 
biological plausibility while maintaining hierarchical structure.
```

---

**Use this guide for all future updates to maintain consistency!**
