# TraceRL (dLLM-RL) Training: Rollout & Optimization Explained

---

## The One Key Difference from d1 / SPG

Before anything else, nail this down:

| | d1 | SPG | TraceRL |
|---|---|---|---|
| **What gets saved after rollout** | Final answer only | Final answer only | Final answer **+ every decoding step** |
| **Optimization unit** | Whole sequence at once | Whole sequence at once | **Each step of the trajectory** |
| **Token probability condition** | Random masked prompt | Block-wise masked answer | **Previously decoded tokens (real prefix)** |



---

## ROLLOUT Phase(same as SPG and d1)
*(Generate real answers and record every step — expensive, done once per round)*

---

**Task:** `"Xiao Ming has 3 apples, ate 1, how many left?"`

**Answer region (initial):** `[ MASK MASK MASK MASK MASK ]`



### Decoding: block-wise dynamic sampling (unmask highest-confidence tokens each step)

| Step | What happens | Token Sequence | Recorded as |
|------|-------------|----------------|-------------|
| Start | All masked | `[prompt] + [MASK MASK MASK │ MASK MASK]` | — |
| Step 1 | In Block 1, top-2 confidence: `"还"` and `"剩"` | `[prompt] + ["还" "剩" MASK │ MASK MASK]` | **τ(1) = {"还", "剩"}** |
| Step 2 | Block 1 still incomplete, unmask `"2"` | `[prompt] + ["还" "剩" "2" │ MASK MASK]` | **τ(2) = {"2"}** |
| Step 3 | Block 1 complete → enter Block 2, unmask `"个"` | `[prompt] + ["还" "剩" "2" │ "个" MASK]` | **τ(3) = {"个"}** |
| Step 4 | Unmask final token `"苹果"` | `[prompt] + ["还" "剩" "2" │ "个" "苹果"]` | **τ(4) = {"苹果"}** |

> ⚠️ **Note:** Within a block, decoding is confidence-based, NOT strictly left-to-right.  
> `"还"` and `"剩"` were decoded together in Step 1 because they had higher confidence than `"2"` at that moment.

---

### After rollout, TWO things are saved:

**1. The final answer:**
```
"还 剩 2 个 苹果"
```

**2. The full trajectory — this is what makes TraceRL different:**
$$\tau_i = (\tau_i(1),\ \tau_i(2),\ \tau_i(3),\ \tau_i(4))$$
$$= (\{"还","剩"\},\ \{"2"\},\ \{"个"\},\ \{"苹果"\})$$

> d1 and SPG throw away the intermediate steps. **TraceRL keeps them all.**

---

### Generate G = 8 different answers, compute reward and advantage:

| Answer | Final Output | Reward rᵢ | Advantage Aᵢ |
|--------|-------------|-----------|--------------|
| τ¹ | "还剩2个苹果" ✓ | r¹ = 1.0 | A¹ = **+0.8** |
| τ² | "还剩5个苹果" ✗ | r² = 0.0 | A² = **−0.8** |
| ... | ... | ... | ... |

$$A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j \quad \text{(standardized)}$$

> Reward is still computed on the **final answer only** (verifiable reward).  
> `τ¹` answers correctly → r = 1. `τ²` answers incorrectly → r = 0.

---

## Shrinkage: Compress the Trajectory Before Training

If we train on every single decoding step separately, a 100-token answer might have 100 steps → 100 forward passes → **training cost explodes**.

**Shrinkage parameter s = 2**: merge every 2 neighboring steps into one training unit.

```
Original trajectory τ:
  τ(1) = {"还","剩"}    τ(2) = {"2"}    τ(3) = {"个"}    τ(4) = {"苹果"}

After shrinkage (s=2) → τˢ:
  τˢ(1) = {"还","剩","2"}      ← steps 1+2 merged
  τˢ(2) = {"个","苹果"}        ← steps 3+4 merged
```

$$|\tau^s_i| = \lceil |\tau_i| / s \rceil \quad \text{→ forward passes reduced by factor } s$$

> This is TraceRL's efficiency trick. s=1 = most faithful to trajectory, most expensive.  
> s=large ≈ similar to d1 (treat whole answer as one chunk), cheap but loses alignment.  
> In practice, **s=8** is used for full-attention models on coding tasks.

---


---

## OPTIMIZATION Phase — With Value Model
*(Assign credit to each step individually)*

---

### Why add a value model?

Without a value model, every token in the trajectory shares the same advantage Aᵢ.  
But intuitively, decoding `"2"` (the key answer) deserves more credit than decoding `"还"` (a grammatical connector).  
The value model makes this distinction.

---

### Step 1: Assign token-wise immediate reward rⱼ

Per paper Section 5.7: **put the full verifiable reward on the last step's tokens; set all earlier tokens' immediate reward to 0.**

```
τ(1) = {"还","剩"}  →  r_还 = 0,  r_剩 = 0
τ(2) = {"2"}        →  r_2  = 0
τ(3) = {"个"}       →  r_个 = 0
τ(4) = {"苹果"}     →  r_苹果 = 1.0   ← full reward lands here
```

---

### Step 2: Aggregate to step-wise reward r★ₜ

$$r_t^\star = \frac{1}{|\tau(t)|}\sum_{j \in \tau(t)} r_j$$

```
r★₁ = (0 + 0) / 2  = 0.0   ← {"还","剩"}
r★₂ = 0 / 1        = 0.0   ← {"2"}
r★₃ = 0 / 1        = 0.0   ← {"个"}
r★₄ = 1.0 / 1      = 1.0   ← {"苹果"}  ← only this step has reward
```

---

### Step 3: Compute step-wise return R★ₜ (backward pass, γ = 1)

$$R_t^\star = r_t^\star + \gamma R_{t+1}^\star, \quad R_{|\tau|+1}^\star = 0$$

```
R★₄ = 1.0 + 0   = 1.0
R★₃ = 0.0 + 1.0 = 1.0    ← future reward propagates back
R★₂ = 0.0 + 1.0 = 1.0
R★₁ = 0.0 + 1.0 = 1.0
```

> Even though early steps have **zero immediate reward**, their **return is positive**  
> because they led to the correct final answer. This is credit assignment.

---

### Step 4: Compute token-wise return Rⱼ

$$R_j = r_j + \gamma R_{t_j+1}^\star$$

where $t_j$ = the step that token $j$ belongs to.

```
R_还   = 0 + R★₂ = 0 + 1.0 = 1.0    (还 is in step 1, next step's return = R★₂)
R_剩   = 0 + R★₂ = 0 + 1.0 = 1.0
R_2    = 0 + R★₃ = 0 + 1.0 = 1.0
R_个   = 0 + R★₄ = 0 + 1.0 = 1.0
R_苹果 = 1 + R★₅ = 1 + 0   = 1.0    (last token, no future return)
```

---

### Step 5: Compute token-wise advantage Aⱼ (with GAE, λ = 1)

$$A_j = (r_j - V_j^{\text{old}}) + \gamma V_{t_j+1}^{\star,\text{old}} + \gamma\lambda A_{t_j+1}^\star$$

Assume the value model estimates:
```
V_还   = 0.3,  V_剩  = 0.3          (early step, low expected value)
V_2    = 0.6                         (model gaining confidence)
V_个   = 0.85
V_苹果 = 1.5   (frozen value overestimates slightly)
V★₁ = (0.3+0.3)/2 = 0.30
V★₂ = 0.60,  V★₃ = 0.85,  V★₄ = 1.5
```

TD residuals:
```
δ★₄ = 1.0 − 1.50 + 0    = −0.50
δ★₃ = 0.0 − 0.85 + 1.50 = +0.65
δ★₂ = 0.0 − 0.60 + 0.85 = +0.25
δ★₁ = 0.0 − 0.30 + 0.60 = +0.30
```

GAE step-wise advantages (λ=1, backward):
```
A★₄ = −0.50
A★₃ = +0.65 + 1×(−0.50) = +0.15
A★₂ = +0.25 + 1×(+0.15) = +0.40
A★₁ = +0.30 + 1×(+0.40) = +0.70
```

Token-wise advantages:
```
A_还   = (0 − 0.3) + 0.60 + 1×0.40 = +0.70   ← same as step 1
A_剩   = (0 − 0.3) + 0.60 + 1×0.40 = +0.70
A_2    = (0 − 0.6) + 0.85 + 1×0.15 = +0.40
A_个   = (0 − 0.85) + 1.50 + 1×(−0.50) = +0.15
A_苹果 = (1 − 1.5) + 0    + 1×0       = −0.50  ← slightly negative (overestimated)
```

> **Now each token has its own advantage**, not the same score for everyone.  
> `"还"` and `"剩"` get +0.70 because they opened a path that led to a good outcome.  
> `"苹果"` gets −0.50 because the value model overestimated it.  
> This is far more informative than the flat A = +0.8 from the no-value-model case.



---

## Block Diffusion Models: Sliced Training (Section 4.3)

For **block-attention models** (e.g., SDAR, TraDo), TraceRL gets a free efficiency bonus.

Each block only needs **one forward pass** to handle all its internal trajectory steps:

```
Trajectory τ with B=3 after rollout:
  Block 1 steps: τ(1)={"还","剩"},  τ(2)={"2"}     ← 2 steps inside Block 1
  Block 2 steps: τ(3)={"个"},       τ(4)={"苹果"}  ← 2 steps inside Block 2

Training:
  Block 1: one forward pass handles τ(1) and τ(2) together (block-attention is parallel)
  Block 2: one forward pass handles τ(3) and τ(4) together

Total: 2 forward passes for 4 steps
```

For full-attention models with the same trajectory, shrinkage s=2 also gives 2 forward passes — but block-attention achieves this naturally via architecture, not as a workaround.

---

## Summary: Three Papers Side by Side

| | d1 | SPG | TraceRL |
|---|---|---|---|
| **Rollout saves** | Final answer | Final answer | Final answer + full trajectory τ |
| **Optimization input** | Random masked prompt + full masked answer | Block-wise MC-sampled partial views of answer | Real decoded prefix at each step |
| **Token prob condition** | Random masked context (not inference-like) | Block-aware partial context | Actual previous decoded tokens ✓ |
| **Advantage** | Sequence-level, shared across all tokens | Sequence-level, shared across all tokens | Sequence-level (no value) or token-wise (with value) |
| **Value model** | No | No | Optional — enables token-wise GAE |
| **Architecture** | Full-attention only | Full-attention only | Full-attention + block-attention |
| **Key insight** | Make GRPO runnable on dLLMs | Fix ELBO/EUBO bias for negative samples | Align training distribution with inference trajectory |

---

## One-Sentence Summary

> **d1:** make RL runnable on dLLMs by approximating log-prob.  
> **SPG:** fix the ELBO bias so negative-advantage samples are penalized correctly.  
> **TraceRL:** stop using random masking entirely — train directly on how the model actually generates.
