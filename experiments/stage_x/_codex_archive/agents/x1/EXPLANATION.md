# Initial candidate

`initial.uncertainty` places a one at BOT in residual channel zero.
`initial.word` maps that indicator to the word information with logit 10.
`initial.stop` gives END logit 1, so it wins after the first word.
All other weights are zero. This is a valid initial artifact, not a successful
chatbot. A broader directly constructed candidate is being developed.
