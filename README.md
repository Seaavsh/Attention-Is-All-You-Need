# Attention-Is-All-You-Need
"Attention Is All You Need" is a famous paper regarding introducing transformers and the begining of GPTs and advancing LLM's. Here is a small project to understand the paper in a hands-on experience. check out my Linkedin post regerding this paper.

The Big Picture: What’s This Toy All About?
This project is a mini-recreation of the Transformer’s self-attention magic, scaled down for fun and learning. The paper revolutionized AI by replacing slow recurrent networks with parallel attention mechanisms, and we’re mimicking that with a tiny sentence ("attention is all you need") and basic math. Our goal? To see how words "attend" to each other, just like the paper’s heatmaps, and visualize it with flair—perfect for a late-night coding party!

Step-by-Step Breakdown: The Code Carnival Ride
Imports and Seed: Setting the Stage
```
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
```
