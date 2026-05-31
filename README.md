# ShroomGPT

Simulating psychedelic effects in LLM inference using REBUS-inspired ODEs for attention sharpness under the modern Hopfield-network interpretation. Future work will explore mechanistic intrepretability techniques, such as graded ablation, to determine if flattening the energy landscape via the inverse temperature will follow a phase-transition. We will formally define a critial point where reasoning is most meaningfully impacted, quantitatively measured by semantic drift and attention entropy. 

![Shroom Forest](shroom-forest.png)

## Mech Interp findings

Attention inverse-temperature flattening of early layers in Llama-3.2-1B substantially blurs those layers' attention and rewrites their value-blended outputs, and — the dominant effect — this perturbation propagates through the residual stream to rewrite the outputs of all downstream layers, which is what degrades coherence. The mechanism is depth-propagation of an early-layer disturbance, not a localized temperature effect.
