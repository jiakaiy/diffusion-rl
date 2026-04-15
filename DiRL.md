# DiRL vs TraceRL: Algorithm-Level Comparison

---

## First: The Objectives Are Almost Identical

DiRL's DiPO objective (paper Equation 6):

$$\mathcal{J}_{policy}(\theta_p) = \mathbb{E}\left[\sum_{i=1}^{G}\sum_{t=1}^{|\tau_i|}\sum_{o_k \in \tau_i(t)} C_\epsilon\!\left(\frac{\pi_{\theta_p}(o_k \mid \tau_i(1{:}t{-}1))}{\pi_{old}(o_k \mid \tau_i(1{:}t{-}1))},\ A_i\right)\!\bigg/|\tau_i(t)|\right] - \beta\,\mathbb{KL}[\pi_\theta \| \pi_{ref}]$$

TraceRL's objective (paper Equation 3) looks almost identical. Both are doing the same thing: **for each decoding step t, conditioned on the real decoded prefix up to step t-1, compute the probability of the tokens decoded at step t.**

So **the algorithmic goal is the same**. The difference lies entirely in: **how efficiently and unbiasedly you compute π_θ(o_k | τ_i(1:t-1)).**

---

## The Core Difference: How Attention Is Computed

This is the heart of it. From Figure 4 in the DiRL paper:

**TraceRL (Figure 4a):**
- Repeats the output twice, then uses a complex irregular attention mask to implement "step t's tokens can attend to all tokens from steps 1 to t-1"
- The mask shape is highly irregular
- **Standard FlashAttention cannot handle this kind of mask** → falls back to vanilla scaled dot-product attention → **slow**

**DiRL (Figure 4b):**
- Repeats both prompt and output blockwise, then reshapes into a more regular mask
- Uses PyTorch's FlexAttention, which accepts fine-grained arbitrary masks
- Same semantics, but the mask is more structured → **FlexAttention handles it efficiently → ~6× faster training**

---

## On "Slices Within Blocks": Both Have Them, But Handle Them Differently

Let's build intuition with a concrete example.

**Setup:**
- Answer: `["still"] ["has"] ["2"] ["apples"] ["left"]`  ← "still has 2 apples left"
- Block size B = 2
- Block 1 = `["still", "has"]`, Block 2 = `["2", "apples"]`, Block 3 = `["left"]`

**Rollout phase (identical for both):**

```
Step 1: Inside Block 1, highest confidence → decode "still"        τ(1) = {"still"}
Step 2: Inside Block 1, decode remaining  → decode "has"           τ(2) = {"has"}    ← Block 1 done
Step 3: Inside Block 2, decode together   → decode "2","apples"    τ(3) = {"2","apples"} ← Block 2 done
Step 4: Inside Block 3, decode            → decode "left"          τ(4) = {"left"}
```

Full trajectory: $\tau = (\{"still"\},\ \{"has"\},\ \{"2","apples"\},\ \{"left"\})$

---

## Optimization Phase: Step-by-Step Comparison

### TraceRL's approach (with shrinkage s=2)

First compress: $\tau^s = (\{"still","has"\},\ \{"2","apples","left"\})$

**Computing probability for τˢ(1) = {"still","has"}, prefix = empty:**

```
Both tokens conditioned on empty prefix, computed together:

π_θ("still" | []) = 0.81
π_θ("has"   | []) = 0.91   ← ⚠️ Approximation here!
                              "has" was actually decoded AFTER "still" in rollout
                              but shrinkage merges the two steps,
                              treating them as if decoded simultaneously
                              → the conditional structure doesn't fully match rollout
```

**Computing probability for τˢ(2) = {"2","apples","left"}, prefix = {"still","has"}:**

```
π_θ("2"      | "still","has") = 0.88
π_θ("apples" | "still","has") = 0.92
π_θ("left"   | "still","has") = 0.95
```

**Attention implementation:** Complex irregular mask → FlashAttention can't handle it → vanilla attention → **slow**

---

### DiRL's approach (no shrinkage, FlexAttention)

DiRL doesn't need shrinkage because FlexAttention can handle fine-grained masks efficiently:

**Step 1: token "still", prefix = empty**

```
Model input: [prompt] + [MASK MASK | MASK MASK | MASK]
                         ↑ Block 1 fully masked (predicting step 1)

π_θ("still" | prompt, []) = 0.81   ← Exact! No random masking anywhere
```

**Step 2: token "has", prefix = {"still"}**

```
Model input: [prompt] + ["still" MASK | MASK MASK | MASK]
                         ↑ "still" is known, rest still masked

π_θ("has" | prompt, "still") = 0.91   ← Exact! "has" genuinely conditioned on "still"
                                           This matches the rollout conditional structure perfectly
```

**Step 3: tokens "2" and "apples", prefix = {"still","has"}**

```
Model input: [prompt] + ["still" "has" | MASK MASK | MASK]
                                         ↑ Block 2 fully masked, but Block 1 is clean

π_θ("2"      | prompt, "still","has") = 0.88   ← Exact! Block 1 fully known
π_θ("apples" | prompt, "still","has") = 0.92   ← Exact!
```

> Here Block 1's content is `["still","has"]` — the tokens **actually decoded during rollout**, not randomly masked.  
> This is precisely where "unbiased" comes from.

**Step 4: token "left", prefix = {"still","has","2","apples"}**

```
Model input: [prompt] + ["still" "has" | "2" "apples" | MASK]

π_θ("left" | prompt, B1, B2) = 0.95   ← Exact!
```

**Attention implementation:** FlexAttention + regular blockwise mask → **~6× faster than TraceRL**

---

## So What Exactly Does "Exact / Unbiased Logits" Mean?

The whole question comes down to: when computing π_θ(o_k | prefix) during optimization, **what is the prefix?**

| Method | What the prefix is | Biased? |
|---|---|---|
| **d1** | Randomly masked prompt (not the real inference prefix) | **Biased** — random masking ≠ real rollout distribution |
| **TraceRL** (with shrinkage) | Real prefix, but neighboring steps are merged → conditional structure is slightly approximated | **Slight approximation** |
| **DiRL** | Fully exact rollout prefix, computed step by step | **Unbiased** |

DiRL achieves "unbiased" for two reasons:

**1. Block-to-block transitions are strictly autoregressive:**  
Block 2 is conditioned on Block 1. Block 1's content comes from the actual rollout — it's what the model genuinely decoded, not a random guess. The condition is exact.

**2. Intra-block denoising probabilities are also exact:**  
Each masked token's prediction probability $p_\theta((b^k_0)_j \mid b^k_t, b^{<k})$ can be read directly from a blockwise forward pass. No multi-step sampling needed to estimate it.

**3. No random masking surrogate at all:**  
d1 needs to randomly mask the prompt to approximate log-prob. DiRL doesn't need this at all — the blockwise structure directly gives you the exact conditional probability for free.

---

## Summary Table

| | d1 | TraceRL | DiRL |
|---|---|---|---|
| **Probability computation** | Random mask + 1 forward pass (ELBO approx.) | Real prefix per step, but steps merged via shrinkage | Real prefix per step, exact, no merging needed |
| **Bias** | High — random masking ≠ inference distribution | Low — slight approximation from shrinkage | None — exact match with rollout |
| **Attention kernel** | Any (full-attention model) | Vanilla attention (FlashAttention incompatible) | FlexAttention (handles fine-grained masks) |
| **Training speed** | Slow (no inference engine) | Baseline | ~6× faster training, ~2.5× overall throughput |
| **Training-inference integration** | None | Checkpoint save/load each step | LMDeploy API server + in-place weight update |
| **Shrinkage needed** | N/A | Yes (to control cost) | No (FlexAttention makes it unnecessary) |

---

## One-Line Summary

> **TraceRL** identified the right direction — use real inference trajectories for RL training — and introduced shrinkage to make it tractable, but the attention implementation is slow.
>
> **DiRL** takes the same core idea, eliminates shrinkage (FlexAttention makes it unnecessary), replaces the slow attention kernel with FlexAttention for 6× speedup, and integrates LMDeploy with online weight updates to eliminate all training-inference IO overhead — turning the concept into a production-grade engineering system.

They are algorithmic siblings. DiRL is the version where someone actually sat down and did the co-design properly.
