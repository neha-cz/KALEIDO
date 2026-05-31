# ShroomGPT

Simulating psychedelic effects in LLM inference using REBUS-inspired ODEs for attention sharpness under the modern Hopfield-network interpretation. Future work will explore mechanistic intrepretability techniques, such as graded ablation, to determine if flattening the energy landscape via the inverse temperature will follow a phase-transition. We will formally define a critial point where reasoning is most meaningfully impacted, quantitatively measured by semantic drift and attention entropy. 

![Shroom Forest](shroom-forest.png)

## Mech Interp findings

All-layer attention inverse-temperature flattening in Llama-3.2-1B produces large associative drift while preserving local grammaticality — a more "coherent-but-loosened" profile than sampling temperature — but it does so without measurably flattening attention entropy and without being statistically separable from temperature on the intended discriminators. 
