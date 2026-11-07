# Initial candidate

One decoder layer, width 32, one head of width one, zero-width MLP, no positions
and no normalization. Attention weights are zero. The residual token indicator
and linear readout implement a one-word reply and termination. Full target
context is accepted structurally; semantic memory is absent.
