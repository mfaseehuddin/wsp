run 1: full K=2 design (plan level n=1 + token level n=48 w/ layout MLP and copy head). dev exact 1.0 first try. Fixed HIC weight 0.3 before (local test: 'hi , i am X . <unknown>' went hello).
local (own phrasings, model.generate): names found only via lexicon+context; "carlos"/"kim" are <unk> in vocab, so tested with in-vocab names. Fixed: self-FOLLOW leak at message end ("grace speaking" copied "speaking") -> FOLLOW/INTRO words get the -3 penalty; "this is urgent ," -> large adjective penalty list; "name" is <unk> so "my name is X" relies on lexicon.
run 2: E_DIM 32->16, D 48->32 (nonzero 444k -> 223k), more keywords/penalties. dev 1.0.
run 3: probe "jade speaking , printer jammed , ticket number 2468" -> exact; level 0 decodes to "" (plan lives in non-token latent dims).
run 4: added ages(slow), online(wifi). dev 1.0. Final.
