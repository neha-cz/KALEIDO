# ShroomGPT

Simulating psychedelic effects in LLM inference using REBUS-inspired ODEs for attention sharpness under the modern Hopfield-network interpretation. Future work will explore mechanistic intrepretability techniques, such as graded ablation, to determine if flattening the energy landscape via the inverse temperature will follow a phase-transition. We will formally define a critial point where reasoning is most meaningfully impacted, quantitatively measured by semantic drift and attention entropy. 

![Shroom Forest](shroom-forest-mech-interp.png)

## Mech Interp findings

Attention inverse-temperature flattening of early layers in Llama-3.2-1B substantially blurs those layers' attention and rewrites their value-blended outputs, and — the dominant effect — this perturbation propagates through the residual stream to rewrite the outputs of all downstream layers, which is what degrades coherence. The mechanism is depth-propagation of an early-layer disturbance, not a localized temperature effect.

This loosened-but-propagating behavior loosely echoes the entropic brain hypothesis: reducing the precision of attention raises the entropy of the network's dynamics and expands the repertoire of representations it explores, producing the elevated associative drift we observe while local grammaticality is initially preserved. 

We frame the result against the entropic-brain account rather than REBUS specifically, because the mechanism diverges from REBUS in two ways. First, REBUS attributes the psychedelic state to relaxed precision of high-level priors, whereas our effect originates in early, low-level layers. Second, REBUS describes a top-down loosening that lets suppressed bottom-up signal through, whereas our effect is a bottom-up cascade — an early-layer perturbation propagating forward through the residual stream. 

The result thus reproduces the entropic-brain signature (precision down, entropy up, expanded state repertoire) without instantiating the REBUS mechanism (high-level prior relaxation, top-down).
