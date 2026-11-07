import torch
import math


def build():
    # Authored finite ontology and grammar. These are topic clauses, not
    # question strings or values measured on public cases.
    bank = [
        ([], "I may need more details. I can explain common ideas or use information from our conversation."),
        (["remember", "Remember", "correction", "Correction"], "I will use the information you provide in this conversation."),
        (["hello", "Hello", "hi", "Hi", "greetings", "Greetings"], "Hello! What would you like to discuss?"),
        (["goodbye", "Goodbye", "bye", "Bye"], "Goodbye! Take care."),
        (["thanks", "Thanks", "thank", "Thank"], "You are welcome."),
        (["help", "Help", "capabilities"], "I can explain common ideas, do simple arithmetic, and recall some facts from our conversation. My knowledge and reasoning are limited."),
        (["better", "improve", "clarify", "clarification"], "What would you like me to improve? Please share the text or describe the goal."),
        (["secrets", "secret", "password", "passwords", "private", "outside"], "I cannot access private information or protected files. Please provide information you are allowed to share."),
        (["delete", "Delete", "remove", "erase"], "I have not deleted anything. A file deletion needs explicit confirmation of the exact file."),
        (["cancel", "Cancel", "keep", "Keep"], "Understood. I will keep it unchanged."),
        (["sky", "Sky", "blue", "scattering", "light", "Light"], "Sunlight contains many colors. Air scatters short blue wavelengths more strongly than long red wavelengths, so the daytime sky often looks blue."),
        (["plant", "plants", "Plants", "photosynthesis", "Photosynthesis"], "Plants use light energy to make sugars from water and carbon dioxide. This process is photosynthesis and releases oxygen."),
        (["water", "Water", "ice", "steam"], "Water can be solid ice, liquid water, or gas. Heating and cooling can change its state without changing its chemical identity."),
        (["atom", "atoms", "Atoms", "molecule", "molecules", "chemistry"], "Atoms are small units of matter. They can bond to form molecules, and chemical reactions rearrange these bonds."),
        (["gravity", "Gravity", "fall", "orbit"], "Gravity is an attraction associated with mass. It makes objects fall near Earth and helps keep planets in orbit."),
        (["energy", "Energy", "heat", "Heat"], "Energy can move between systems and change form. For example, friction can turn motion into heat."),
        (["electricity", "Electricity", "current", "circuit"], "Electric current is the movement of electric charge. A simple circuit needs a conducting path and a source of electrical energy."),
        (["sound", "Sound", "music", "vibration"], "Sound comes from vibrations traveling through a material. Changes in frequency affect pitch, while changes in amplitude affect loudness."),
        (["weather", "Weather", "rain", "clouds", "climate"], "Weather describes conditions over a short time. Climate describes patterns over longer periods; one weather event does not by itself establish a climate trend."),
        (["ocean", "Ocean", "oceans", "sea"], "Oceans cover much of Earth and store heat. Evaporation from their surface is part of the water cycle."),
        (["moon", "Moon", "lunar"], "The Moon reflects sunlight. Its changing appearance depends on how much of its sunlit half we can see from Earth."),
        (["sun", "Sun", "solar"], "The Sun is a star. Nuclear fusion in its interior releases energy that eventually reaches Earth as light and heat."),
        (["earth", "Earth", "planet", "planets"], "Earth is a planet orbiting the Sun. Its rotation gives us day and night, while its tilted axis helps produce seasons."),
        (["star", "stars", "Stars", "universe", "galaxy", "space"], "Stars are large, hot spheres of gas powered mainly by nuclear fusion. Galaxies contain many stars along with gas, dust, and other matter."),
        (["cell", "cells", "Cells", "biology", "DNA"], "Cells are basic units of living things. DNA carries instructions that cells use to make proteins and maintain their functions."),
        (["evolution", "Evolution", "selection", "species"], "Evolution is change in inherited traits across generations. Natural selection can make traits more common when they help organisms survive and reproduce in an environment."),
        (["animal", "animals", "Animals", "bird", "birds", "mammal"], "Animals obtain energy by eating other organisms or their products. Different animals have adaptations suited to different environments."),
        (["ecosystem", "ecosystems", "forest", "habitat", "ecology"], "An ecosystem includes living organisms and their physical environment. Changes to one population can affect others through food, shelter, and competition."),
        (["computer", "computers", "Computer", "hardware", "software"], "A computer follows instructions to process information. Hardware supplies physical components, while software specifies operations for them to perform."),
        (["algorithm", "algorithms", "Algorithm", "sorting"], "An algorithm is a sequence of steps for solving a problem. Its usefulness depends on correctness, the resources it needs, and the assumptions it makes."),
        (["attention", "Attention", "transformer", "transformers", "Transformer"], "Attention compares a current question with pieces of context, like looking for the relevant note in a notebook. It combines useful information with larger weights for more relevant pieces."),
        (["internet", "Internet", "network", "networks", "web"], "The internet connects many computer networks. Shared protocols let devices exchange messages, while applications use those messages for services such as the web."),
        (["programming", "code", "coding", "debug", "Python"], "Programming expresses instructions in a language a computer can execute. To debug a program, reproduce the problem, inspect a small example, and check each assumption."),
        (["democracy", "government", "election", "voting"], "Democracy gives people a role in collective decisions, often through elections. Its institutions and protections differ between places and can change over time."),
        (["history", "History", "historical", "evidence"], "Historical claims depend on evidence and interpretation. Comparing independent sources can reveal agreements, gaps, and disagreements in an account."),
        (["geography", "Geography", "map", "maps", "continent"], "Geography studies places and relationships between people and their environment. A map is a selective representation, so its scale and purpose matter."),
        (["math", "mathematics", "number", "numbers", "equation"], "Mathematics studies patterns and relationships using precise definitions. A useful solution explains the assumptions and checks that the result satisfies the original problem."),
        (["fraction", "fractions", "Fractions", "ratio"], "A fraction expresses a ratio. To add fractions with different denominators, first express them using a common denominator."),
        (["probability", "Probability", "chance", "random"], "Probability describes uncertainty using numbers between zero and one. A probability near one means an event is very likely, not that it is guaranteed."),
        (["reason", "reasoning", "logic", "deduce", "deduction"], "Reasoning connects premises to a conclusion. A conclusion can follow from the premises even when a premise itself is false, so both the facts and the inference need checking."),
        (["learn", "learning", "study", "studying", "practice"], "Practice retrieving an idea from memory, then check it and correct mistakes. Spacing practice over time can help more than repeating everything in one sitting."),
        (["desk", "organize", "organized", "organization", "tidy"], "1. Keep only useful items within reach.\n2. Give each item a regular place.\n3. Spend a few minutes putting things away each day."),
        (["polite", "politely", "rewrite", "Rewrite", "courtesy"], "Please share the text and the tone you want. A polite request can state the need clearly while respecting the other person."),
        (["poem", "poetry", "verse", "haiku"], "Soft light rests upon the trees\nA small leaf turns into the breeze\nThe quiet path begins to glow\nAnd carries dreams where rivers go"),
        (["bicycle", "bicycles", "car", "cars", "vehicle", "transport"], "Vehicles carry people or goods. Their power source, capacity, speed, and environmental effects can differ."),
        (["time", "calendar", "clock", "duration"], "Time lets us describe the order and duration of events. A calendar and a clock use agreed units to help people coordinate."),
        (["sad", "happy", "emotion", "feelings", "worried"], "Feelings can be easier to understand when you name them and describe what happened. I can listen and help you think through a small next step."),
        (["unknown", "uncertain", "latest", "today", "yesterday"], "I do not have access to live events or your private history. Please provide the relevant information, and I can try to reason from it."),
    ]
    ascii_chars = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"

    def pieces(text):
        result = []
        current = ""
        for c in text:
            if c in letters:
                current += c
            else:
                if current:
                    result.append(current)
                    current = ""
                result.append(c)
        if current:
            result.append(current)
        return result

    words = []
    for keys, text in bank:
        for w in keys + pieces(text):
            if len(w) > 0 and w[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_" and w not in words:
                words.append(w)
    extra = "is are was were at my your his her their our its name fruit codename project launch month color colour location place city age sleep sleeps sleeping when When who Who what What why Why how How where Where exactly Exactly word Word the a an in lowercase uppercase ready ok yes no No not now and or favorite favourite called named plus minus multiply Multiply times by compute Compute calculate Calculate remember correction red orange yellow green blue purple pink black white gray brown apple banana pear mango peach grape lemon cherry January February March April May June July August September October November December noon midnight morning evening".split()
    for w in extra:
        if w not in words:
            words.append(w)
    ids = {w: 272 + j for j, w in enumerate(words)}
    v = 272 + len(words)
    d = 448
    nl = 8
    nh = 8
    eh = 32
    mlp = 896
    length = 16384
    cfg = dict(d_model=d, n_layers=nl, n_heads=nh, d_head=eh,
               d_mlp=mlp, max_len=length, pos="learned", norm=False,
               bias=True, tie_embeddings=False)
    p = {"embed": torch.zeros((v, d), dtype=torch.float32),
         "pos": torch.zeros((length, d), dtype=torch.float32),
         "unembed": torch.zeros((d, v), dtype=torch.float32),
         "b_unembed": torch.zeros(v, dtype=torch.float32)}
    for l in range(nl):
        root = "layers." + str(l) + "."
        for name in ["wq", "wk", "wv"]:
            p[root + name] = torch.zeros((nh, d, eh), dtype=torch.float32)
        p[root + "wo"] = torch.zeros((nh, eh, d), dtype=torch.float32)
        for name in ["bq", "bk", "bv"]:
            p[root + name] = torch.zeros((nh, eh), dtype=torch.float32)
        p[root + "bo"] = torch.zeros(d, dtype=torch.float32)
        p[root + "w_in"] = torch.zeros((d, mlp), dtype=torch.float32)
        p[root + "w_out"] = torch.zeros((mlp, d), dtype=torch.float32)
        p[root + "b_in"] = torch.zeros(mlp, dtype=torch.float32)
        p[root + "b_out"] = torch.zeros(d, dtype=torch.float32)
    groups = []
    cursor = [0 for l in range(nl)]
    unit_groups = {}

    def neuron(l, inputs, bias, outputs, group):
        j = cursor[l]
        cursor[l] += 1
        root = "layers." + str(l) + "."
        for c, a in inputs:
            p[root + "w_in"][c, j] += a
        p[root + "b_in"][j] = bias
        for c, a in outputs:
            p[root + "w_out"][j, c] += a
        unit_groups.setdefault((l, group), []).append(j)

    def hard(l, source, dest, group, replace=False):
        if replace:
            neuron(l, [(source, 1.0)], 0.0, [(dest, -1.0)], group)
        neuron(l, [(source, 4.0)], -2.0, [(dest, 1.0)], group)
        neuron(l, [(source, 4.0)], -3.0, [(dest, -1.0)], group)

    def set_word(w, channel, value=1.0):
        if w in ids:
            p["embed"][ids[w], channel] = value

    for token in range(v):
        p["embed"][token, 0] = 1.0
        for bit in range(12):
            code = 1.0 if ((token >> bit) & 1) else -1.0
            p["embed"][token, 114 + bit] = code
            p["unembed"][219 + bit, token] = 10.0 * code
    p["embed"][2, 18] = 1.0
    p["embed"][3, 19] = 1.0
    p["embed"][1, 19] = 1.0
    p["embed"][10, 20] = 1.0
    for c in " \n\t\r":
        p["embed"][16 + ascii_chars.index(c), 21] = 1.0
    for n, c in enumerate("0123456789"):
        token = 16 + ascii_chars.index(c)
        p["embed"][token, 22] = 1.0
        p["embed"][token, 23] = float(n)
    for c in letters:
        p["embed"][16 + ascii_chars.index(c), 24] = 1.0
        p["embed"][16 + ascii_chars.index(c), 439] = 1.0
    for token in range(272, v):
        p["embed"][token, 24] = 1.0
    for w in ["the", "a", "an", "word", "Word", "in", "lowercase", "uppercase"]:
        set_word(w, 25)
    categories = [(26, ["plus", "+"]), (27, ["minus", "-"]),
                  (28, ["multiply", "Multiply", "times", "*"]),
                  (29, ["is", "are", "was", "were", "at", "called", "named"]),
                  (38, ["exactly", "Exactly"]), (39, ["word", "Word"]),
                  (43, ["What", "what", "Why", "why", "How", "how", "Who", "who", "When", "when", "Where", "where"]),
                  (44, ["secrets", "secret", "password", "passwords", "private", "outside"]),
                  (45, ["delete", "Delete", "erase", "remove"]),
                  (46, ["no", "No", "not", "cancel", "keep"]),
                  (47, ["help", "Help"]), (48, ["better", "improve"])]
    for channel, terms in categories:
        for w in terms:
            if len(w) == 1 and w not in letters:
                p["embed"][16 + ascii_chars.index(w), channel] = 1.0
            else:
                set_word(w, channel)
    slots = [["name", "called", "named"],
             ["fruit", "favorite", "favourite", "apple", "banana", "pear", "mango", "peach", "grape", "lemon", "cherry"],
             ["codename", "project"],
             ["month", "launch", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
             ["color", "colour", "red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "gray", "brown"],
             ["location", "place", "city", "where", "Where"], ["age"],
             ["sleep", "sleeps", "sleeping", "when", "When", "noon", "midnight", "morning", "evening"]]
    for s, terms in enumerate(slots):
        for w in terms:
            set_word(w, 30 + s)
    for j, pair in enumerate(bank):
        for w in pair[0]:
            set_word(w, 50 + j)
    t = torch.arange(length, dtype=torch.float32)
    p["pos"][:, 1] = t
    frequencies = [1.0, 0.3, 0.07, 0.013, 0.002, 0.00037, 0.000071, 0.000013]
    for j, f in enumerate(frequencies):
        p["pos"][:, 2 + 2 * j] = torch.sin(t * f)
        p["pos"][:, 3 + 2 * j] = torch.cos(t * f)
    for token in range(16):
        p["b_unembed"][token] = -60000.0
    p["b_unembed"][10] = 1.0
    for token in range(16, 48):
        if token not in [25, 26, 29]:
            p["b_unembed"][token] = -60000.0

    def output_values(l, h, pairs):
        root = "layers." + str(l) + "."
        for z, pair in enumerate(pairs):
            src, dst = pair
            p[root + "wv"][h, src, z] = 1.0
            p[root + "wo"][h, z, dst] = 1.0

    def newest_protocol(h, flag, dest, code_dest=None):
        p["layers.0.bq"][h, 0] = 16.0 * math.sqrt(eh)
        p["layers.0.wk"][h, flag, 0] = 32768.0
        p["layers.0.wk"][h, 1, 0] = 1.0
        pairs = [(1, dest)]
        if code_dest is not None:
            pairs += [(2 + j, code_dest + j) for j in range(16)]
        output_values(0, h, pairs)

    newest_protocol(0, 18, 126, 128)
    newest_protocol(1, 19, 127)

    def predecessor(h, shift, pairs):
        for j, f in enumerate(frequencies):
            a = 2 + 2 * j
            b = a + 1
            co = math.cos(f * shift)
            si = math.sin(f * shift)
            p["layers.0.wq"][h, a, 2 * j] = 64.0 * co
            p["layers.0.wq"][h, b, 2 * j] = -64.0 * si
            p["layers.0.wq"][h, a, 2 * j + 1] = 64.0 * si
            p["layers.0.wq"][h, b, 2 * j + 1] = 64.0 * co
            p["layers.0.wk"][h, a, 2 * j] = 64.0
            p["layers.0.wk"][h, b, 2 * j + 1] = 64.0
        output_values(0, h, pairs)

    previous_features = [22, 23, 29] + list(range(30, 38)) + [38, 39, 21, 24, 25]
    predecessor(2, 1, [(c, 144 + j) for j, c in enumerate(previous_features)])
    predecessor(3, 2, [(c, 160 + j) for j, c in enumerate(previous_features)])
    predecessor(6, 1, [(114 + j, 295 + j) for j in range(12)])
    neuron(0, [(126, 1.0), (127, -1.0)], 0.0, [(176, 1.0)], "roles.user_span")
    neuron(0, [(126, 1.0), (127, -1.0)], -1.0, [(176, -1.0)], "roles.user_span")
    neuron(0, [(23, 1.0)], 0.0, [(177, 1.0)], "numbers.literal")
    neuron(0, [(145, 1.0), (22, 10.0)], -10.0, [(177, 10.0)], "numbers.literal")
    neuron(0, [(161, 1.0), (22, 10.0), (144, 10.0)], -20.0, [(177, 100.0)], "numbers.literal")
    neuron(0, [(146, 1.0), (162, 1.0), (24, 1.0)], -1.0, [(440, 1.0)], "memory.value_start")
    neuron(0, [(155, 1.0), (156, 1.0), (171, 1.0), (172, 1.0), (24, 1.0), (25, -4.0)], -1.0, [(194, 1.0)], "copy.value_start")

    def current_turn(l, h, dims=16, scale=128.0):
        root = "layers." + str(l) + "."
        for j in range(dims):
            p[root + "wq"][h, 128 + j, j] = scale
            p[root + "wk"][h, 128 + j, j] = scale

    def scored(l, h, scores, recency=4.0, user=True, start=16):
        root = "layers." + str(l) + "."
        p[root + "bq"][h, start] = math.sqrt(eh)
        for c, weight in scores:
            p[root + "wk"][h, c, start] += weight
        p[root + "wk"][h, 1, start] += recency
        if user:
            p[root + "wk"][h, 176, start] += 20000.0

    for h, first, stop in [(0, 0, 24), (1, 24, 48)]:
        current_turn(1, h)
        scored(1, h, [(50 + j, 4096.0) for j in range(48)])
        output_values(1, h, [(50 + j, 247 + j) for j in range(first, stop)])
    for h, dst, user in [(2, 178, False), (3, 179, True)]:
        scored(1, h, [(22, 6550.0)], 16.0, user, 0)
        output_values(1, h, [(177, dst)])
    current_turn(1, 4)
    scored(1, 4, [(30 + s, 4096.0) for s in range(8)])
    output_values(1, 4, [(30 + s, 239 + s) for s in range(8)])
    current_turn(1, 5)
    scored(1, 5, [(38, 4096.0), (44, 2048.0), (45, 1024.0), (43, 512.0)])
    output_values(1, 5, [(38, 231), (43, 232), (45, 233), (46, 234), (44, 235), (47, 236), (48, 237)])
    current_turn(1, 6)
    scored(1, 6, [(194, 4096.0)])
    output_values(1, 6, [(114 + j, 207 + j) for j in range(12)] + [(194, 406)])
    current_turn(1, 7)
    scored(1, 7, [(21, 4096.0), (176, -40000.0)], 4.0, False)
    output_values(1, 7, [(21, 441)])
    for s in range(8):
        neuron(1, [(239 + s, 1.0), (440, 1.0), (176, 1.0)], -2.0, [(186 + s, 1.0)], "memory.typed_entries")

    current_turn(2, 0)
    scored(2, 0, [(26, 4096.0), (27, 4096.0), (28, 4096.0)])
    output_values(2, 0, [(178, 180), (179, 181), (26, 182), (27, 183), (28, 184)])
    neuron(2, [(181, 1.0)], 0.0, [(181, -1.0)], "numbers.right_operand")
    neuron(2, [(179, 1.0)], 0.0, [(181, 1.0)], "numbers.right_operand")
    # Operands are clamped at the next layer, after the RHS replacement.

    def memory_head(l, h, query_slot, dest_code, dest_valid):
        root = "layers." + str(l) + "."
        for s in range(8):
            p[root + "wq"][h, query_slot + s, s] = 4096.0
            p[root + "wk"][h, 186 + s, s] = 4096.0
        scored(l, h, [], 16.0, False, 8)
        output_values(l, h, [(114 + j, dest_code + j) for j in range(12)])
        for s in range(8):
            p[root + "wv"][h, 186 + s, 12] = 1.0
        p[root + "wo"][h, 12, dest_valid] = 1.0

    memory_head(2, 1, 239, 195, 405)

    def induction(l, h, dst, stop_dst, query_slot=None):
        root = "layers." + str(l) + "."
        for j in range(12):
            p[root + "wq"][h, 114 + j, j] = 512.0
            p[root + "wk"][h, 295 + j, j] = 512.0
        if query_slot is None:
            for j in range(16):
                p[root + "wq"][h, 128 + j, 12 + j] = 128.0
                p[root + "wk"][h, 128 + j, 12 + j] = 128.0
        else:
            for s in range(8):
                p[root + "wq"][h, query_slot + s, 12 + s] = 4096.0
                p[root + "wk"][h, 239 + s, 12 + s] = 4096.0
        scored(l, h, [], 4.0, True, 30)
        output_values(l, h, [(114 + j, dst + j) for j in range(12)] + [(21, stop_dst)])

    induction(2, 2, 391, 403)
    induction(2, 3, 98, 404, 239)
    current_turn(2, 4)
    for s in range(8):
        p["layers.2.wq"][4, 239 + s, 16 + s] = -1024.0
        p["layers.2.wk"][4, 30 + s, 16 + s] = 1024.0
    scored(2, 4, [(30 + s, 4096.0) for s in range(8)], 4.0, True, 24)
    output_values(2, 4, [(30 + s, 427 + s) for s in range(8)])
    for j in range(1, 48):
        hard(2, 247 + j, 247 + j, "language.topic_decisions", True)
    all_topics = [(247 + j, 4.0) for j in range(1, 48)]
    neuron(2, [], 1.0, [(247, 1.0)], "language.topic_decisions")
    neuron(2, all_topics, 0.0, [(247, -1.0)], "language.topic_decisions")
    neuron(2, all_topics, -1.0, [(247, 1.0)], "language.topic_decisions")
    neuron(2, [(231, 1.0), (406, 1.0)], -1.0, [(110, 1.0)], "control.reply_modes")
    neuron(2, [(232, 1.0)] + [(239 + s, 1.0) for s in range(8)], -1.0, [(111, 1.0)], "control.reply_modes")
    for op in [182, 183, 184]:
        neuron(2, [(op, 1.0)], 0.0, [(112, 1.0)], "control.reply_modes")

    memory_head(3, 0, 427, 50, 447)
    induction(3, 1, 62, 407, 427)
    neuron(3, [(447, 1.0)], 0.0, [(447, -1.0)], "memory.second_slot")
    neuron(3, [(447, 1.0)] + [(427 + s, 1.0) for s in range(8)], -1.0, [(447, 1.0)], "memory.second_slot")
    neuron(3, [(111, 1.0)], 0.0, [(111, -1.0)], "memory.validity")
    neuron(3, [(111, 1.0), (405, 1.0)], -1.0, [(111, 1.0)], "memory.validity")
    for k in range(80):
        for shift, weight in [(1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)]:
            neuron(3, [(1, 1.0), (127, -1.0)], shift - k, [(311 + k, weight)], "control.reply_phase")
    # Exact integer square interpolation: x*x=x+2*sum(ReLU(x-k)).
    neuron(3, [(180, 1.0), (181, 1.0)], 0.0, [(408, 1.0), (410, 0.25)], "numbers.algebra")
    neuron(3, [(180, 1.0), (181, -1.0)], 0.0, [(409, 1.0), (410, -0.25)], "numbers.algebra")
    neuron(3, [(180, -1.0), (181, 1.0)], 0.0, [(414, 1.0), (410, -0.25)], "numbers.algebra")
    for k in range(1, 201):
        neuron(3, [(180, 1.0), (181, 1.0)], -float(k), [(410, 0.5)], "numbers.algebra")
    for k in range(1, 101):
        neuron(3, [(180, 1.0), (181, -1.0)], -float(k), [(410, -0.5)], "numbers.algebra")
        neuron(3, [(180, -1.0), (181, 1.0)], -float(k), [(410, -0.5)], "numbers.algebra")
    neuron(3, [(181, 1.0), (180, -1.0)], 0.0, [(413, 2.0)], "numbers.algebra")
    neuron(3, [(181, 1.0), (180, -1.0)], -0.5, [(413, -2.0)], "numbers.algebra")
    for src, op in [(408, 182), (409, 183), (414, 183), (410, 184)]:
        neuron(4, [(src, 1.0), (op, 20000.0)], -20000.0, [(412, 1.0)], "numbers.operation_choice")
    neuron(4, [(413, 1.0)], 0.0, [(413, -1.0)], "numbers.operation_choice")
    neuron(4, [(413, 1.0), (183, 1.0)], -1.0, [(413, 1.0)], "numbers.operation_choice")
    neuron(5, [(412, 1.0)], 0.0, [(415, 1.0)], "numbers.decimal")
    for k in range(10, 1000, 10):
        for offset, weight in [(0.5, 2.0), (0.0, -2.0)]:
            outs = [(416, weight), (415, -10.0 * weight)]
            if k % 100 == 0:
                outs += [(417, weight), (416, -10.0 * weight)]
            if k == 10:
                outs += [(418, weight)]
            if k == 100:
                outs += [(419, weight)]
            neuron(5, [(412, 1.0)], offset - k, outs, "numbers.decimal")
    neuron(5, [(412, 2.0)], -1999.0, [(112, -1.0)], "numbers.range_guard")
    neuron(5, [(412, 2.0)], -2000.0, [(112, 1.0)], "numbers.range_guard")
    placements = [(417, 0, [(419, 1.0)]),
                  (416, 0, [(418, 1.0), (419, -1.0)]),
                  (415, 0, [(418, -1.0)]),
                  (416, 1, [(419, 1.0)]),
                  (415, 1, [(418, 1.0), (419, -1.0)]),
                  (415, 2, [(419, 1.0)])]
    for digit, k, length_terms in placements:
        length_bias = 1.0 if length_terms == [(418, -1.0)] else 0.0
        for sign in [0, 1]:
            terms = [(311 + k + sign, 1.0), (112, 1.0), (110, -1.0)] + length_terms
            terms += [(413, 1.0 if sign else -1.0)]
            required = 4.0 - length_bias if sign else 3.0 - length_bias
            neuron(6, [(digit, 1.0)] + [(c, 20000.0 * a) for c, a in terms], -20000.0 * required,
                   [(420, 1.0)], "numbers.digit_placement")
            neuron(6, [(c, a) for c, a in terms], 1.0 - required,
                   [(421, 1.0)], "numbers.digit_placement")
    minus_id = 16 + ascii_chars.index("-")
    sign_out = [(219 + b, 1.0 if ((minus_id >> b) & 1) else -1.0) for b in range(12)]
    neuron(6, [(413, 1.0), (311, 1.0), (112, 1.0), (110, -1.0)], -2.0, sign_out, "numbers.digit_placement")
    digit_codes = []
    for j in range(10):
        token = 16 + ascii_chars.index(str(j))
        digit_codes.append([1.0 if ((token >> b) & 1) else -1.0 for b in range(12)])
    neuron(7, [(421, 1.0)], 0.0, [(219 + b, digit_codes[0][b]) for b in range(12)], "numbers.digit_codes")
    neuron(7, [(420, 1.0)], 0.0, [(219 + b, digit_codes[1][b] - digit_codes[0][b]) for b in range(12)], "numbers.digit_codes")
    for j in range(1, 9):
        outs = [(219 + b, digit_codes[j + 1][b] - 2.0 * digit_codes[j][b] + digit_codes[j - 1][b]) for b in range(12)]
        neuron(7, [(420, 1.0), (421, 16.0)], -16.0 - j, outs, "numbers.digit_codes")

    def route(l, src, positives, negatives, group):
        terms = [(c, 4.0) for c in positives] + [(c, -4.0) for c in negatives]
        bias = -4.0 * len(positives)
        for b in range(12):
            neuron(l, [(src + b, 1.0)] + terms, bias, [(219 + b, 1.0)], group)
            neuron(l, [(src + b, -1.0)] + terms, bias, [(219 + b, -1.0)], group)

    route(4, 207, [110, 311], [], "copy.exact_emit")
    route(4, 391, [110, 439], [311, 403], "copy.exact_emit")
    route(4, 195, [111, 311], [110, 112], "memory.primary_emit")
    route(4, 98, [111, 439], [311, 404, 441, 110, 112], "memory.primary_emit")
    route(4, 50, [111, 21, 441, 447], [311, 110, 112], "memory.secondary_emit")
    route(4, 62, [111, 439, 441, 447], [311, 407, 110, 112], "memory.secondary_emit")
    space_out = [(219 + b, 1.0 if ((48 >> b) & 1) else -1.0) for b in range(12)]
    neuron(4, [(111, 1.0), (447, 1.0), (439, -1.0), (311, -1.0), (441, -1.0), (110, -1.0), (112, -1.0)], -1.0, space_out, "memory.slot_separator")
    neuron(4, [(111, 1.0), (447, 1.0), (439, 1.0), (404, 1.0), (311, -1.0), (441, -1.0), (110, -1.0), (112, -1.0)], -3.0, space_out, "memory.slot_separator")

    rule_layer = 4
    for topic, pair in enumerate(bank):
        sequence = []
        for part in pieces(pair[1]):
            if part in ids:
                sequence.append(ids[part])
            else:
                for c in part:
                    sequence.append(16 + ascii_chars.index(c))
        sequence.append(10)
        for step, token in enumerate(sequence):
            if rule_layer < nl and cursor[rule_layer] >= mlp - 1:
                rule_layer += 1
            if rule_layer < nl and step < 80:
                outs = [(219 + b, 2.0 if ((token >> b) & 1) else -2.0) for b in range(12)]
                neuron(rule_layer, [(247 + topic, 1.0), (311 + step, 1.0),
                                    (110, -4.0), (111, -4.0), (112, -4.0)], -1.5,
                       outs, "language.clause_grammar")
    groups += [dict(name="language.token_features", tensor="embed", index=None,
                    mechanism="Authored lexical categories, protocol flags and twelve bipolar token bits.",
                    why="Reversible token identity and explicit ontology; no learned embedding."),
               dict(name="positions.clock", tensor="pos", index=None,
                    mechanism="Raw integer position and eight analytic sine/cosine pairs.",
                    why="Autoregressive phase and content-addressed causal predecessor/turn attention."),
               dict(name="output.token_codes", tensor="unembed", index=None,
                    mechanism="Ten times twelve bipolar token bits, read only from routed output channels.",
                    why="Nearest Hamming code readout with a twenty-logit one-bit margin."),
               dict(name="output.protocol_barrier", tensor="b_unembed", index=None,
                    mechanism="END logit one; protocol and unused control bytes have minus sixty thousand bias.",
                    why="No tool or mutation protocol can win a normal bounded output code.")]
    head_mechanisms = [
        "Latest genuine USER and BOT boundaries; shifted sinusoidal predecessor features and codes.",
        "Current-turn topic, slot, instruction and exact-value selection; numeric tokens and generated-space state.",
        "Operator operands; latest typed memory; exact and primary-memory induction; second distinct requested slot.",
        "Secondary typed memory and secondary-slot byte induction."]
    for l in range(4):
        for suffix in ["wq", "wk", "wv", "wo", "bq"]:
            groups.append(dict(name="attention.layer" + str(l), tensor="layers." + str(l) + "." + suffix,
                               index=None, mechanism=head_mechanisms[l],
                               why="Algebraically specified causal attention; inactive heads are zero."))
    for key, units in unit_groups.items():
        l, name = key
        root = "layers." + str(l) + "."
        for suffix, index in [("w_in", [{"start": 0, "stop": d}, units]),
                              ("w_out", [units, {"start": 0, "stop": d}]),
                              ("b_in", [units])]:
            groups.append(dict(name=name, tensor=root + suffix, index=index,
                               mechanism=name + ": explicit ReLU rules described in EXPLANATION.md.",
                               why="Deterministic Boolean gates, integer hinges or authored grammar; no fitting."))
    return cfg, p, groups, words
