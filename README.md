# ShroomGPT

Probing psychedelic-like effects in LLM inference by flattening attention sharpness via the inverse temperature, motivated by the modern Hopfield-network interpretation and the entropic brain hypothesis. Using graded ablation and mechanistic interpretability, we find that flattening early layers blurs their attention but loosens reasoning chiefly through depth-propagation of the perturbation along the residual stream, not a localized temperature effect or a clean energy-landscape phase transition. Rather than a sharp critical point, we observe a graded altered-but-coherent band, quantified by semantic drift, attention entropy, and per-layer output shift. The bottom-up cascade reproduces the entropic-brain signature which posits top-down relaxation of high-level priors.

![Shroom Forest](shroom-forest-mech-interp.png)

## Mech Interp findings

Attention inverse-temperature flattening of early layers in Llama-3.2-1B substantially blurs those layers' attention and rewrites their value-blended outputs, and — the dominant effect — this perturbation propagates through the residual stream to rewrite the outputs of all downstream layers, which is what degrades coherence. The mechanism is depth-propagation of an early-layer disturbance, not a localized temperature effect.

This loosened-but-propagating behavior loosely echoes the entropic brain hypothesis: reducing the precision of attention raises the entropy of the network's dynamics and expands the repertoire of representations it explores, producing the elevated associative drift we observe while local grammaticality is initially preserved. 

We frame the result against the entropic-brain account rather than REBUS specifically, because the mechanism diverges from REBUS in two ways. First, REBUS attributes the psychedelic state to relaxed precision of high-level priors, whereas our effect originates in early, low-level layers. Second, REBUS describes a top-down loosening that lets suppressed bottom-up signal through, whereas our effect is a bottom-up cascade — an early-layer perturbation propagating forward through the residual stream. 

The result thus reproduces the entropic-brain signature (precision down, entropy up, expanded state repertoire) without instantiating the REBUS mechanism (high-level prior relaxation, top-down).
