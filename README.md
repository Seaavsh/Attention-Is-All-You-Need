# Attention-Is-All-You-Need
"Attention Is All You Need" is a famous paper regarding introducing transformers and the begining of GPTs and advancing LLM's. Here is a small project to understand the paper in a hands-on experience. check out my Linkedin post regerding this paper.
Read the full paper here: https://arxiv.org/abs/1706.03762

## The Big Picture: What’s This Toy All About?
This project is a mini recreation of the Transformer’s self-attention magic, scaled down for fun and learning. The paper revolutionized AI by replacing slow recurrent networks with parallel attention mechanisms, and we’re mimicking that with a tiny sentence ("attention is all you need") and basic math. Our goal? To see how words "attend" to each other, just like the paper’s heatmaps, and visualize. Perfect for a late night coding party!

## Step-by-Step Breakdown: The Code Carnival Ride
### Imports and Seed: Setting the Stage
```
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
```
We’re summoning NumPy and Matplotlib like trusty sidekicks, and seed(42) (the Hitchhiker’s Guide to the Galaxy nod!) ensures our random numbers play nice every time.
Positional Encoding: The Wave Dance Party (Section 3.5)
### Positional Encoding: The Wave Dance Party (Section 3.5)
```
def positional_encoding(max_pos, d_model):
    pe = np.zeros((max_pos, d_model))
    position = np.arange(0, max_pos, dtype=float)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * - (np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
```
Math: Positional encoding adds order since Transformers ditch recurrence. The formula uses sine and cosine waves:

$$PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i / d_{\text{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)$$

Here, pos is the word’s position (0 to n-1), d_model is 8 (our toy dimension), and the denominator (10,000) sets a max wavelength. This creates unique "fingerprints" for each position, blending with embeddings. Imagine words doing a synchronized dance , sine for even beats, cosine for odd, grooving to a rhythm that tells "attention" it’s first and "need" it’s last!
### Fun Detective Takeaway
Imagine Figure 1 in the paper as a party floor plan! The positional encoding is the DJ’s playlist. Sine and Cosine tracks that keep every guest (word) in sync, no matter how many show up. The stacked layers are dance floors, and the attention arrows are guests mingling globally. Scalability? Just invite more dancers, the playlist adapts without missing a beat, just as the paper’s design intends!
Paper Link: The left side of Figure 1 shows the input feeding into the encoder. While positional encoding isn’t labeled, the uniform stacking of layers suggests a pre-processed input ready for parallel attention , Section 3.5’s contribution.

### Scaled Dot-Product Attention: The Party Mixer (Section 3.2.1)
```
def scaled_dot_product_attention(q, k, v):
    matmul_qk = np.dot(q, k.T)
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    output = np.dot(attention_weights, v)
    return output, attention_weights
```

Math: This is the heart! For queries $$Q$$, keys $$K$$, and values $$V$$ (all $$n \times d_k$$ matrices), the attention score is:
$${Attention}(Q, K, V) = {softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

$$QK^T$$ computes dot products (compatibility scores).
Divide by $$\sqrt{d_k}$$ (here, $$\sqrt{8} \approx 2.83$$) stabilizes gradients (paper’s key insight).
Softmax normalizes: $${softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$$ (numerically stable version).
Output is a weighted sum of $$V$$.

Picture a cocktail party where you (query) ask around (keys), score who’s worth listening to (dot products), and mix their stories (values) into a perfect drink. The scaling prevents a math hangover!
Paper Link:Figure 2 , our single-head version simplifies the multi-head frenzy (h=8 in the paper).
### The Sentence Setup: Our Cast of Characters
```
sentence = "attention is all you need".split()
n = len(sentence)
d_model = 8
embeddings = np.random.randn(n, d_model)
pe = positional_encoding(n, d_model)
x = embeddings + pe
q = k = v = x
```
Math: We create random embeddings ($$n \times d_{model}$$), add positional encodings, and set $$Q = K = V = X$$ for self-attention. In reality, $$Q, K, V$$ are linear projections (paper’s Equation 1), but our toy skips that for simplicity.
It’s a self love fest. Each word checks itself in the mirror (diagonal attention) while gossiping with pals (off-diagonals). Try " Fight for your dreams "  !!!
### Running the Show and Printing Weights
```
output, attention_weights = scaled_dot_product_attention(q, k, v)
print("Sentence:", sentence)
print("Attention Weights (rounded):")
print(np.round(attention_weights, decimals=3))
```
Math: " attention_weights " is an $$n \times n$$ matrix where $$w_{ij}$$ shows how word $$i$$ attends to $$j$$. Sum of each row = 1 (softmax property).
It’s the guest list with scores. Who’s the VIP (high weights) and who’s just background noise?
## Visualization: 
### Heatmaps(Figure 3 to 5 Style)
```
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(attention_weights, cmap='viridis', interpolation='nearest')
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(sentence, rotation=45, ha='right')
ax.set_yticklabels(sentence)
for i in range(n):
    for j in range(n):
        text = ax.text(j, i, f'{attention_weights[i, j]:.3f}', ha='center', va='center', color='w' if attention_weights[i, j] > 0.5 else 'black')
ax.set_title("Self-Attention Heatmap: Word Connections (Toy Model)")
plt.colorbar(im, ax=ax, label='Attention Score')
plt.tight_layout()
plt.show()
```
Math: We simulate two heads by perturbing $$Q, K, V$$ slightly (paper uses linear projections and h=8 heads). Each head’s "attention_weights" varies, showing diverse focus.
It’s a detective duo! One head might track "is" to "all," another "you" to "need" . Like Figure 4’s "its" resolution or Figure 5’s syntax mapping.
Paper Link: Echoes multi-head visuals (Figure 2 right, Figure 5), where heads specialize (e.g., delimiter vs. binding).

### Bar Chart of Self-Attention
```
fig2, ax2 = plt.subplots(figsize=(10, 5))
self_attention = np.diag(attention_weights)
ax2.bar(sentence, self_attention, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'], edgecolor='black')
ax2.set_title("Self-Attention Strength per Word")
ax2.set_ylabel("Attention Weight")
ax2.set_ylim(0, 1)
for i, v in enumerate(self_attention):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
```

### Running the Show: Your Late Night Experiment

Setup: 
```
pip install numpy matplotlib
````
Run in Jupyter or Python.
Output: A matrix of weights (e.g., diagonal strong, off-diagonals weaker) and two heatmaps. one single-head, one multi-head style. Tweak sentence to " Fight for your dreams" and watch the patterns shift!

### Why It’s Awesome (Paper Parallels)
Our toy skips the paper’s 512 $$d_{model}$$ and 6-layer stacks, but the core $$O(n^2 \cdot d)$$ attention scales like theirs (Table 1).
Heatmaps with text annotations mimic Figure 3-5’s clarity, showing how "attention" links to "need" (context resolution).

## Your Turn: Join the Carnival!
Run this, snap your heatmaps, and share on LinkedIn! Tweak the code add heads, adjust $$d_{model}$$. Share your feedback with me andddddd ....
let’s geek out!

Siavash Sharifnezhad

Email: siavash.s7.79@gmail.com

Linkedin: https://www.linkedin.com/in/sharifnezhad
