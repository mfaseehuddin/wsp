"""pw100: a hand-written rule-program chat transformer for wsp-pure. Every weight is set by the code below.

Family "transformer", norm "none" (all arithmetic stays linear and exact), learned positions (pos_emb carries
CONST=1, POS=p and a sinusoid code), tied embeddings.  See EXPLANATION.md for the full story.
"""
import math

import torch

from wsp.pure import transformer as TF
from wsp.pure.harness import ids, special

V = 151936
DO = 192            # token identity code size (OWN subspace; also the output accumulator, since tied)
NL, NH, DH = 11, 16, 64
MAXCTX = 8192
C_OUT = 30.0        # emission strength of a template token
MG = 20000.0        # gate constant: pre-activation shift that switches a bank off
SQ = math.sqrt(DH)
IM_START, IM_END = special("<|im_start|>"), special("<|im_end|>")

# ============================================================== knowledge (hand-written)
# Intent concepts.  "^W" = only the sentence-start capitalised token "W"; "=W" = lowercase no-space allowed too.
INTENT_WORDS = {
    "QW": "=what =who =whom =whose =where =when =why =how =which ^Tell ^Explain ^Describe ^Give ^Name ^List ^Define "
          "^Say ^Compare tell explain describe define",
    "HARM": "bomb bombs explosive explosives weapon weapons gun guns poison poisoning kill killing murder hack hacking "
            "steal stealing rob robbing meth cocaine heroin malware ransomware phishing counterfeit shoplift "
            "shoplifting kidnap stalk stalking arson launder laundering suicide overdose terrorist illegal",
    "IT": "=it =this =that",
    "ACT": "=make =book =fix =do =send =buy =order =improve =finish =handle =schedule =cancel =reserve",
    "PRIV": "password passwords thinking yesterday pin secret diary",
    "MY": "=my", "YOUR": "=your yourself", "YOU": "=you", "CALL": "=call", "NAMEW": "=name", "WHO": "=who",
    "ARE": "=are", "HOW": "=how", "DOING": "=doing feeling going", "HELP": "=help assist", "CAN": "=can =could",
    "THINGS": "things kinds kind sort able abilities capabilities",
    "THANKS": "=thanks =thank thx grateful appreciate appreciated",
    "BYE": "=bye =goodbye farewell", "GREET": "=hello =hi =hey greetings morning afternoon evening =howdy hiya",
    "ENC": "encourage encouraging encouragement motivate motivation motivating inspire inspiring cheer uplifting",
    "JOKE": "joke jokes funny", "CORR": "correction correct actually update updated changed moved instead",
    "REM": "remember note memorize also", "POLITE": "politely polite rewrite rephrase",
    "SUMM": "summarize summarise summary", "POEM": "poem poems verse haiku rhyme",
    "STORY": "story sentences paragraph tale", "TIPS": "tips advice suggestions",
    "WRITE": "=write =tell =make =create =compose",
    "OP_TIMES": "times multiply multiplied multiplying product",
    "OP_DIV": "divided divide dividing quotient",
    "OP_MINUS": "minus subtract subtracted subtracting gave ate lost spent left remain remaining difference",
    "OP_PLUS": "plus add adds added adding sum total together bought found received gained more",
    "OP_BIG": "bigger larger greater higher maximum",
    "OP_SMALL": "smaller lesser fewer lower minimum less",
    "THAT": "answer result =that =it",
    "LARGEST": "largest biggest", "SMALLEST": "smallest tiniest", "HOTTEST": "hottest", "FASTEST": "fastest quickest",
    "TALLEST": "tallest highest", "LONGEST": "longest", "RED": "red", "BLUE": "blue", "YELLOW": "yellow",
    "WHITE": "white", "PRIMARY": "primary",
}
SPECIAL_TOKENS = {"QMARK": ["?", " ?", "??", "?!", "?\n"], "IM": ["I'm|-1", " I'm|-1", " am"], "POSS": ["Lena's|-1"],
                  "OP_TIMES": ["*", " *", " x", "×", " ×"], "OP_DIV": ["/", " /", "÷", " ÷"],
                  "OP_MINUS": ["-", " -", "−", " −"], "OP_PLUS": ["+", " +"]}
OPS = ["OP_TIMES", "OP_DIV", "OP_MINUS", "OP_PLUS", "OP_BIG", "OP_SMALL"]      # priority order
NUMWORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen twenty").split()
NUMWORDS = [(w, i) for i, w in enumerate(NUMWORDS)] + [("thirty", 30), ("forty", 40), ("fifty", 50), ("sixty", 60),
                                                       ("seventy", 70), ("eighty", 80), ("ninety", 90), ("hundred", 100)]
INTENT_WORDS.update({"EXACTLY": "exactly only", "REPEAT": "repeat echo"})

# ---- stage 2: sentence / span copying.  Token lists are written as text; only single-token forms are used.
BOUNDARY_TXT = [".", "!", "?", ":", ";", ".\n", "!\n", "?\n", ":\n", ".\n\n", "...", " .", " !", " ?", " :", "\n",
                "\n\n"]
CLOSE_TXT = [".", "!", ".\n", "!\n", ".\n\n", "...", " .", " !"]           # statement closers (+ <|im_end|>)
COLON_TXT = [":", " :", ":\n"]
NL_TXT = ["\n", "\n\n"]
TERM_TXT = BOUNDARY_TXT + [",", " ,"] + [" " + w for w in
            "and but or from in at to with by of for on who which that so because now today here too also again "
            "please was".split()]
TRIG1_TXT = ["I'm|-1", " I'm|-1", " called", " named", " word"]   # span starts after these
TIS_TXT, TNAMEW_TXT = [" is"], [" name", "name", " Name", "Name"]  # "name is"
TME_TXT, TCALLW_TXT = [" me"], [" call", "call", " Call", "Call"]  # "call me"
TAM_TXT, TIW_TXT = [" am"], ["I", " I"]                            # "I am"
STOPWORDS = ("a an the and or but if then so of to in on at by for with from about as into over under is are was "
             "were be been being am do does did have has had i me my mine myself we our you your yours he him his "
             "she her it its they them their this that these those what which who whom whose when where why how "
             "there here all any each some no not can could would should will shall may might must also just very "
             "too please tell give say reply write make let know think s m re ve ll d one now really much many "
             "remember correction update actually").split()
NOTNAME = ("fine good great ok okay sorry tired happy sad here back not so very a an the just going looking trying "
           "thinking doing feeling glad well also really still sure new from in at on with to bored hungry busy "
           "excited worried confused lost done ready home sick afraid scared interested curious working learning "
           "reading writing studying planning wondering asking calling having getting making taking coming leaving "
           "staying living moving hoping about over all only always never sometimes too pretty quite kind sort "
           "your my his her their our its this that what it is was are").split()
SUBST = {
    "RECALL": [(" my", " your"), ("My", "your"), (" My", " your"), (" I", " you"), ("I", "you"), ("'m", "'re"),
               (" me", " you"), (" am", " are"), (" mine", " yours"), (" myself", " yourself")],
    "POLITE": [(" " + w.capitalize(), " " + w) for w in
               "give send tell bring show make get do fix call stop write clean pass close open help find take put "
               "turn finish move leave come go answer reply check wait buy book order print share".split()]
              + [(".", "?"), ("!", "?")],
}
INTRO_SUFFIX = "! I'm Pip, a small chat program. How can I help you today?"
# slot plans: (name, None, None, prefix, copy type, suffix, substitution set, condition)
#   type N = name span after a trigger, E = sentence after ':', S = best-matching earlier user statement
SLOT_PLANS = {
    "EXACT_N": ("EXACT_N", None, None, "", "N", "", None, "EXACT"),
    "STORY_N": ("STORY_N", None, None, "", "N", " was a curious little creature who loved to explore. One sunny "
                "morning, it found a hidden path that led to a field full of flowers.", None, "STORY"),
    "SUMM_E": ("SUMM_E", None, None, "In short:", "E", "", None, "SUMM"),
    "POLITE_E": ("POLITE_E", None, None, "Could you please", "E", " Thank you!", "POLITE", "POLITE"),
    "REPEAT_E": ("REPEAT_E", None, None, "", "E", "", None, "REPEAT"),
    "RECALL": ("RECALL", None, None, "You told me that", "S", "", "RECALL", "RECALL"),
    "INTRO_N1": ("INTRO_N1", None, None, "Nice to meet you,", "N", INTRO_SUFFIX, None, "INTRO_IM"),
    "INTRO_N2": ("INTRO_N2", None, None, "Nice to meet you,", "N", INTRO_SUFFIX, None, "INTRO_NAME"),
}

# Entity cards, specific first (earlier = higher priority when several are present).  words | card
ENTITIES = """
feather feathers | A kilogram of feathers and a kilogram of iron weigh the same. Both weigh exactly one kilogram.
France | The capital of France is Paris. France is a country in Western Europe, and people there speak French.
Germany | The capital of Germany is Berlin. Germany is a country in Central Europe, and people there speak German.
Italy | The capital of Italy is Rome. Italy is a country in Southern Europe, and people there speak Italian.
Spain | The capital of Spain is Madrid. Spain is a country in Southern Europe, and people there speak Spanish.
Portugal | The capital of Portugal is Lisbon. Portugal is a country in Southern Europe, and people there speak Portuguese.
England Britain UK | The capital of England and of the United Kingdom is London. People there speak English.
Ireland | The capital of Ireland is Dublin. Ireland is an island country in Europe, and people there speak English and Irish.
Scotland | The capital of Scotland is Edinburgh. Scotland is part of the United Kingdom.
Japan | The capital of Japan is Tokyo. Japan is an island country in East Asia, and people there speak Japanese.
China | The capital of China is Beijing. China is a large country in East Asia, and people there speak Chinese.
India | The capital of India is New Delhi. India is a large country in South Asia, and people there speak Hindi, English and many other languages.
Russia | The capital of Russia is Moscow. Russia is the largest country in the world by area, and people there speak Russian.
Canada | The capital of Canada is Ottawa. Canada is a large country in North America, and people there speak English and French.
America USA | The capital of the United States is Washington, D.C. The United States is a country in North America, and most people there speak English.
Mexico | The capital of Mexico is Mexico City. Mexico is a country in North America, and people there speak Spanish.
Brazil | The capital of Brazil is Brasilia. Brazil is the largest country in South America, and people there speak Portuguese.
Argentina | The capital of Argentina is Buenos Aires. Argentina is a country in South America, and people there speak Spanish.
Australia | The capital of Australia is Canberra. Australia is both a country and a continent, and people there speak English.
Egypt | The capital of Egypt is Cairo. Egypt is a country in North Africa, famous for its pyramids, and people there speak Arabic.
Kenya | The capital of Kenya is Nairobi. Kenya is a country in East Africa, and people there speak Swahili and English.
Nigeria | The capital of Nigeria is Abuja. Nigeria is a country in West Africa, and its official language is English.
Greece | The capital of Greece is Athens. Greece is a country in Southern Europe, and people there speak Greek.
Sweden | The capital of Sweden is Stockholm. Sweden is a country in Northern Europe, and people there speak Swedish.
Norway | The capital of Norway is Oslo. Norway is a country in Northern Europe, and people there speak Norwegian.
Denmark | The capital of Denmark is Copenhagen. Denmark is a country in Northern Europe, and people there speak Danish.
Finland | The capital of Finland is Helsinki. Finland is a country in Northern Europe, and people there speak Finnish and Swedish.
Poland | The capital of Poland is Warsaw. Poland is a country in Central Europe, and people there speak Polish.
Netherlands Holland | The capital of the Netherlands is Amsterdam. It is a country in Western Europe, and people there speak Dutch.
Belgium | The capital of Belgium is Brussels. People in Belgium speak Dutch, French and German.
Switzerland | The capital of Switzerland is Bern. People in Switzerland speak German, French, Italian and Romansh.
Austria | The capital of Austria is Vienna. Austria is a country in Central Europe, and people there speak German.
Hungary | The capital of Hungary is Budapest. Hungary is a country in Central Europe, and people there speak Hungarian.
Thailand | The capital of Thailand is Bangkok. Thailand is a country in Southeast Asia, and people there speak Thai.
Vietnam | The capital of Vietnam is Hanoi. Vietnam is a country in Southeast Asia, and people there speak Vietnamese.
Korea | The capital of South Korea is Seoul, and the capital of North Korea is Pyongyang. People there speak Korean.
Indonesia | The capital of Indonesia is Jakarta. Indonesia is an island country in Southeast Asia, and people there speak Indonesian.
Philippines | The capital of the Philippines is Manila. People there speak Filipino and English.
Peru | The capital of Peru is Lima. Peru is a country in South America, and people there speak Spanish.
Chile | The capital of Chile is Santiago. Chile is a long, narrow country in South America, and people there speak Spanish.
Colombia | The capital of Colombia is Bogota. Colombia is a country in South America, and people there speak Spanish.
Cuba | The capital of Cuba is Havana. Cuba is an island country in the Caribbean, and people there speak Spanish.
Iran | The capital of Iran is Tehran. Iran is a country in Western Asia, and people there speak Persian.
Pakistan | The capital of Pakistan is Islamabad. Pakistan is a country in South Asia, and people there speak Urdu and English.
Ukraine | The capital of Ukraine is Kyiv. Ukraine is a country in Eastern Europe, and people there speak Ukrainian.
Morocco | The capital of Morocco is Rabat. Morocco is a country in North Africa, and people there speak Arabic and Berber.
Ethiopia | The capital of Ethiopia is Addis Ababa. Ethiopia is a country in East Africa, and people there speak Amharic.
Iceland | The capital of Iceland is Reykjavik. Iceland is an island country in the North Atlantic.
Zealand | The capital of New Zealand is Wellington. People in New Zealand speak English and Maori.
Paris | Paris is the capital of France. It is famous for the Eiffel Tower and the Louvre museum.
London | London is the capital of England and of the United Kingdom. It is on the River Thames.
Tokyo | Tokyo is the capital of Japan and one of the largest cities in the world.
Rome | Rome is the capital of Italy. It is famous for the Colosseum and its long history.
Africa | Africa is the second largest continent. It has 54 countries, and the Sahara Desert and the Nile River are there.
Europe | Europe is a continent with about 44 countries, including France, Germany and Italy.
Asia | Asia is the largest continent, both in area and in population. China and India are in Asia.
Antarctica | Antarctica is the coldest continent. It is covered in ice, and penguins live there.
continent continents | There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America and South America. Asia is the largest.
Mercury | Mercury is the smallest planet and the closest planet to the Sun.
Venus | Venus is the second planet from the Sun and the hottest planet in our solar system.
Mars | Mars is the fourth planet from the Sun. It is called the Red Planet because of its red, rusty soil.
Jupiter | Jupiter is the largest planet in our solar system. It is a gas giant with a giant storm called the Great Red Spot.
Saturn | Saturn is the sixth planet from the Sun, famous for its bright rings.
Uranus | Uranus is the seventh planet from the Sun. It is an ice giant that spins on its side.
Neptune | Neptune is the eighth and farthest planet from the Sun. It is a cold, windy ice giant.
Pluto | Pluto is a dwarf planet at the edge of our solar system.
moon | The Moon is Earth's only natural satellite. It goes around Earth about once a month, and Neil Armstrong first walked on it in 1969.
sun | The Sun is the star at the center of our solar system. It is a huge ball of hot gas that gives Earth light and heat.
Earth | Earth is the third planet from the Sun and the only planet known to have life. About 71 percent of its surface is covered by water.
planet planets | There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune. Jupiter is the largest.
star stars | Stars are huge balls of hot gas that make their own light. The Sun is the closest star to Earth.
galaxy | A galaxy is a huge group of stars. We live in the Milky Way galaxy.
astronaut astronauts | An astronaut is a person trained to travel into space. Neil Armstrong was the first person to walk on the Moon.
spider spiders | A spider has eight legs. Spiders are arachnids, not insects, and many spin silk webs to catch food.
octopus octopuses | An octopus has eight arms, three hearts and blue blood. It is a clever sea animal that can change color.
whale whales | A whale is a mammal, not a fish. Whales breathe air, and the blue whale is the largest animal that has ever lived.
dolphin dolphins | A dolphin is a smart mammal that lives in the sea. Dolphins breathe air and talk with clicks and whistles.
shark sharks | A shark is a fish whose skeleton is made of cartilage. Sharks have lived in the oceans for millions of years.
bee bees | Bees are insects that make honey and wax. They help plants grow by carrying pollen between flowers.
ant ants | Ants are small insects that live in large groups called colonies. They can carry many times their own weight.
butterfly butterflies | A butterfly is an insect that starts life as a caterpillar. It has four colorful wings.
snake snakes | A snake is a reptile with no legs. Snakes smell with their tongues.
frog frogs | A frog is an amphibian. It starts life as a tadpole in water and jumps with its strong back legs.
penguin penguins | A penguin is a bird that cannot fly, but it swims very well. Most penguins live in the Southern Hemisphere.
owl owls | An owl is a bird that hunts at night. It can turn its head very far around.
bat bats | A bat is the only mammal that can truly fly. Many bats find food at night by listening to echoes.
elephant elephants | An elephant is the largest land animal. It has a long trunk, big ears and tusks, and it eats plants.
giraffe giraffes | A giraffe is the tallest animal. Its long neck helps it eat leaves from tall trees.
lion lions | A lion is a big cat that lives in groups called prides. It is often called the king of the jungle.
tiger tigers | A tiger is the largest wild cat. It has orange fur with black stripes.
cheetah cheetahs | A cheetah is the fastest land animal. It can run about 100 kilometers per hour in short bursts.
horse horses | A horse is a large animal with four legs and hooves. People ride horses, and horses eat grass and hay.
cow cows | A cow is a farm animal. Cows eat grass, drink water, and give us milk.
pig pigs | A pig is a smart farm animal. Pigs roll in mud to stay cool.
sheep | Sheep are farm animals with wool. They eat grass, and their wool is used to make clothes.
chicken chickens hen hens | A chicken is a farm bird. Hens lay eggs.
duck ducks | A duck is a bird that swims and flies. It has webbed feet and a flat bill.
dog dogs puppy puppies | A dog is a loyal pet and a mammal. Dogs have four legs and a great sense of smell.
cat cats kitten kittens | A cat is a small furry pet and a mammal. Cats have four legs and sharp claws, and they sleep a lot.
rabbit rabbits bunny | A rabbit is a small mammal with long ears. Rabbits eat grass and vegetables.
mouse mice | A mouse is a small rodent with a long tail.
kangaroo kangaroos | A kangaroo is a mammal from Australia. It hops on its strong back legs and carries its baby in a pouch.
koala koalas | A koala is a mammal from Australia that eats eucalyptus leaves and sleeps most of the day.
panda pandas | A giant panda is a bear from China. It eats mostly bamboo.
bear bears | A bear is a large mammal with thick fur. Many bears sleep through the winter.
monkey monkeys | A monkey is a primate that lives in trees. Monkeys use their hands and tails to climb.
camel camels | A camel is a desert animal. Its hump stores fat, so it can go a long time without food or water.
zebra zebras | A zebra is a wild horse from Africa with black and white stripes.
wolf wolves | A wolf is a wild animal related to dogs. Wolves live and hunt in packs.
fox foxes | A fox is a small wild animal with a bushy tail. Foxes are clever and often hunt at night.
deer | A deer is a gentle animal that eats plants. Male deer grow antlers.
eagle eagles | An eagle is a large bird of prey with very sharp eyes.
parrot parrots | A parrot is a colorful bird that can copy sounds and words.
crocodile crocodiles | A crocodile is a large reptile with strong jaws that lives near water.
turtle turtles tortoise | A turtle is a reptile with a hard shell. Some turtles live for more than 100 years.
dinosaur dinosaurs | Dinosaurs were reptiles that lived millions of years ago. They died out about 66 million years ago.
insect insects | An insect has six legs and three body parts. Ants, bees and butterflies are insects.
bird birds | Birds are animals with feathers, wings and beaks. Most birds can fly, and they lay eggs.
fish | Fish live in water and breathe through gills. Most fish have fins and scales.
mammal mammals | Mammals are warm-blooded animals with hair or fur, and they feed their babies milk. Dogs, whales and humans are mammals.
reptile reptiles | Reptiles are cold-blooded animals with scales, like snakes, lizards and turtles.
photosynthesis plant plants | Plants use sunlight to make their own food. This is called photosynthesis: they turn water and carbon dioxide into sugar and give off oxygen.
tree trees | Trees are tall plants with wooden trunks. They make oxygen and give us wood and fruit.
flower flowers | Flowers are the colorful parts of plants. They make seeds, and bees visit them for nectar.
grass | Grass is a green plant. Cows, sheep and horses eat it.
ice | Ice is frozen water. It floats on water because it is less dense than liquid water.
fire | Fire is the heat and light made when something burns. It needs fuel, oxygen and heat.
oxygen | Oxygen is a gas in the air that people and animals need to breathe. Plants release it.
sky | Sunlight is made of many colors, and the air scatters blue light the most. That is why the sky looks blue.
rain | Rain is water that falls from clouds. It forms when water vapor cools into drops.
snow | Snow is frozen water that falls from clouds as ice crystals when it is very cold.
cloud clouds | Clouds are made of tiny water drops or ice crystals floating in the air.
rainbow rainbows | A rainbow appears when sunlight shines through raindrops. It has seven colors: red, orange, yellow, green, blue, indigo and violet.
wind | Wind is moving air. It happens because the Sun heats the Earth unevenly.
lightning thunder | Lightning is a giant spark of electricity in a storm, and thunder is the sound it makes.
volcano volcanoes | A volcano is an opening in the Earth where hot melted rock, called lava, comes out.
earthquake earthquakes | An earthquake is a shaking of the ground caused by moving plates deep inside the Earth.
Pacific | The Pacific Ocean is the largest and deepest ocean on Earth.
ocean oceans | There are five oceans, and the Pacific Ocean is the largest and deepest. Oceans cover about 71 percent of Earth.
Nile | The Nile is one of the longest rivers in the world. It flows through Egypt into the Mediterranean Sea.
Amazon | The Amazon is a huge river and rainforest in South America. It carries more water than any other river.
river rivers | A river is fresh water flowing toward a sea or a lake. The Nile and the Amazon are the longest rivers.
Everest | Mount Everest is the tallest mountain on Earth, about 8,849 meters high, in the Himalayas.
mountain mountains | A mountain is a very high landform. Mount Everest is the tallest mountain on Earth.
Sahara | The Sahara is the largest hot desert in the world. It is in North Africa.
desert deserts | A desert is a very dry place with little rain. The Sahara is the largest hot desert.
forest forests rainforest | A forest is a large area covered with trees. Forests are home to many animals and plants.
gravity | Gravity is the force that pulls things toward each other. It keeps us on the ground and the Moon around Earth. Isaac Newton described it.
sound | Sound is a vibration that travels through air, water or solids. It cannot travel through empty space.
electricity | Electricity is the flow of tiny charged particles called electrons. It powers lights and machines.
magnet magnets | A magnet pulls on iron and steel. Every magnet has a north pole and a south pole.
atom atoms | An atom is the tiny building block of all matter. It has protons, neutrons and electrons.
cell cells | A cell is the smallest unit of life. All living things are made of cells.
DNA | DNA is the molecule that carries the instructions for how living things grow and work.
blood | Blood carries oxygen and food around the body. The heart pumps it.
heart | The heart is a muscle that pumps blood around the body. At rest it beats about 60 to 100 times a minute.
brain | The brain controls the body. It helps us think, feel, move and remember.
lung lungs | The lungs take in oxygen from the air and let out carbon dioxide when we breathe.
bone bones skeleton | An adult human has 206 bones. Bones hold the body up and protect the organs.
tooth teeth | Adults usually have 32 teeth, and children have 20 baby teeth.
banana bananas | A banana is a long yellow fruit. It is sweet, soft and full of potassium.
apple apples | An apple is a crunchy fruit that can be red, green or yellow. Apples grow on trees.
lemon lemons | A lemon is a sour yellow citrus fruit.
strawberry strawberries | A strawberry is a sweet red fruit with tiny seeds on the outside.
grape grapes | Grapes are small round fruits that grow in bunches. They are used to make raisins and juice.
carrot carrots | A carrot is an orange root vegetable. It is rich in vitamin A.
potato potatoes | A potato is a starchy vegetable that grows underground.
tomato tomatoes | A tomato is a red fruit that is often cooked like a vegetable.
bread | Bread is a food made from flour, water and yeast, and then baked.
milk | Milk is a white drink that comes from cows, goats and other mammals. It has calcium for strong bones.
cheese | Cheese is a food made from milk.
egg eggs | Eggs are laid by birds such as chickens. They are a good source of protein.
rice | Rice is a grain, and it is a main food for billions of people.
honey | Honey is a sweet food that bees make from the nectar of flowers.
chocolate | Chocolate is a sweet food made from roasted cacao beans.
coffee | Coffee is a drink made from roasted coffee beans. It has caffeine.
tea | Tea is a drink made by soaking tea leaves in hot water.
vegetable vegetables | Vegetables are plant foods like carrots, broccoli and potatoes. They are full of vitamins.
fruit fruits | Fruits are the sweet parts of plants that hold seeds. A banana is yellow, a strawberry is red, and an orange is orange.
Monday | Monday comes after Sunday and before Tuesday. It is the first day of the work week.
Tuesday | Tuesday comes after Monday and before Wednesday.
Wednesday | Wednesday comes after Tuesday and before Thursday. It is the middle of the work week.
Thursday | Thursday comes after Wednesday and before Friday.
Friday | Friday comes after Thursday and before Saturday. It is the last day of the work week.
Saturday | Saturday comes after Friday and before Sunday. It is part of the weekend.
Sunday | Sunday comes after Saturday and before Monday. It is part of the weekend.
January | January is the first month of the year. It has 31 days.
February | February is the second month of the year. It has 28 days, or 29 in a leap year.
April | April is the fourth month of the year. It has 30 days.
June | June is the sixth month of the year. It has 30 days.
July | July is the seventh month of the year. It has 31 days.
September | September is the ninth month of the year. It has 30 days.
October | October is the tenth month of the year. It has 31 days.
November | November is the eleventh month of the year. It has 30 days.
December | December is the twelfth and last month of the year. It has 31 days.
weekend weekends | The weekend is Saturday and Sunday.
week weeks | A week has seven days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday.
year years | A year has 12 months and 365 days, or 366 days in a leap year.
month months | There are 12 months in a year. Most months have 30 or 31 days, and February has 28 or 29.
hour hours | An hour has 60 minutes.
minute minutes | A minute has 60 seconds.
decade decades | A decade is 10 years.
century centuries | A century is 100 years.
season seasons | There are four seasons: spring, summer, autumn (also called fall) and winter.
triangle triangles | A triangle has three sides and three corners. Its angles add up to 180 degrees.
square squares | A square has four equal sides and four right angles.
rectangle rectangles | A rectangle has four sides and four right angles, and its opposite sides are equal.
circle circles | A circle is a round shape. Every point on it is the same distance from the center.
pentagon | A pentagon has five sides.
hexagon | A hexagon has six sides. Honeycomb cells are hexagons.
octagon | An octagon has eight sides, like a stop sign.
dozen | A dozen is 12.
prime | A prime number has exactly two factors, 1 and itself. The first prime numbers are 2, 3, 5, 7 and 11.
noun nouns | A noun is a word for a person, place, thing or idea. For example, "dog" is a noun.
verb verbs | A verb is a word for an action or a state. For example, "run" is a verb.
adjective adjectives | An adjective is a word that describes a noun. For example, "blue" in "a blue car" is an adjective.
adverb adverbs | An adverb is a word that describes a verb, like "quickly" in "she ran quickly".
pronoun pronouns | A pronoun is a word that takes the place of a noun, like "he", "she" or "it".
vowel vowels | The vowels in English are a, e, i, o and u.
alphabet | The English alphabet has 26 letters, from A to Z.
Shakespeare Hamlet Romeo Juliet Macbeth | William Shakespeare was an English playwright. He wrote Romeo and Juliet, Hamlet and Macbeth.
Einstein relativity | Albert Einstein was a physicist famous for the theory of relativity and the equation E = mc^2.
Newton | Isaac Newton was an English scientist who described gravity and the laws of motion.
telephone | Alexander Graham Bell is credited with inventing the telephone in 1876.
lightbulb bulb | Thomas Edison made the first practical light bulb in 1879.
airplane airplanes aeroplane | The Wright brothers made the first powered airplane flight in 1903.
Mona Lisa | Leonardo da Vinci painted the Mona Lisa.
Columbus | Christopher Columbus sailed across the Atlantic Ocean to the Americas in 1492.
internet | The internet is a global network that connects computers all around the world.
computer computers | A computer is a machine that follows instructions to store and process information.
robot robots | A robot is a machine that can do tasks by itself by following a program.
piano | A piano is a musical instrument with 88 black and white keys.
guitar | A guitar is a string instrument, usually with six strings.
soccer football | Soccer, also called football, is played by two teams of eleven players who try to kick a ball into the other team's goal.
Olympics Olympic | The Olympic Games are a big international sports event held every four years.
bicycle bicycles bike bikes | A bicycle has two wheels and pedals. You ride it by pushing the pedals with your feet.
car cars | A car is a vehicle with four wheels and an engine.
water | Water is made of hydrogen and oxygen (H2O). At sea level it boils at 100 degrees Celsius and freezes at 0 degrees Celsius.
air | Air is a mix of gases, mostly nitrogen and oxygen. We need it to breathe.
light | Light is energy that lets us see. It travels about 300,000 kilometers per second, faster than anything else.
color colors colour colours | The primary colors of paint are red, yellow and blue. Mixing blue and yellow makes green.
animal animals | Animals are living things that move and eat food. There are millions of kinds, from tiny insects to huge whales.
hot | Hot means having a high temperature. The opposite of hot is cold.
cold | Cold means having a low temperature. The opposite of cold is hot.
big large | Big means large in size. A synonym for big is "large", and the opposite of big is small.
small little | Small means little in size. A synonym for small is "little", and the opposite of small is big.
fast quick | Fast means moving quickly. A synonym for fast is "quick", and the opposite of fast is slow.
slow | Slow means not fast. The opposite of slow is fast.
happy | Happy means feeling good. A synonym for happy is "joyful", and the opposite of happy is sad.
sad | Sad means feeling unhappy. A synonym for sad is "unhappy", and the opposite of sad is happy.
good | A synonym for good is "great", and the opposite of good is bad.
bad | A synonym for bad is "awful", and the opposite of bad is good.
tall | Tall means high from bottom to top. The opposite of tall is short.
short | Short means not long or not tall. The opposite of short is long or tall.
young | Young means not old. The opposite of young is old.
dark | Dark means having little light. The opposite of dark is light or bright.
heavy | Heavy means weighing a lot. The opposite of heavy is light.
hard | Hard means firm or difficult. The opposite of hard is soft or easy.
soft | Soft means not hard. The opposite of soft is hard.
easy | Easy means not difficult. A synonym for easy is "simple", and the opposite of easy is hard.
difficult | Difficult means hard to do. A synonym for difficult is "hard", and the opposite is easy.
rich | Rich means having a lot of money. The opposite of rich is poor.
poor | Poor means having little money. The opposite of poor is rich.
strong | Strong means having a lot of power. The opposite of strong is weak.
weak | Weak means having little power. The opposite of weak is strong.
loud | Loud means making a lot of noise. The opposite of loud is quiet.
quiet | Quiet means making little noise. The opposite of quiet is loud.
early | Early means before the usual time. The opposite of early is late.
late | Late means after the usual time. The opposite of late is early.
full | Full means holding as much as possible. The opposite of full is empty.
empty | Empty means holding nothing. The opposite of empty is full.
wet | Wet means covered with water. The opposite of wet is dry.
dry | Dry means not wet. The opposite of dry is wet.
clean | Clean means not dirty. The opposite of clean is dirty.
dirty | Dirty means not clean. The opposite of dirty is clean.
smart clever | Smart means quick to learn and understand. A synonym for smart is "intelligent".
brave | Brave means ready to face danger. A synonym for brave is "courageous", and the opposite is cowardly.
angry | Angry means feeling mad. A synonym for angry is "mad", and the opposite is calm.
beautiful pretty | Beautiful means very nice to look at. A synonym for beautiful is "lovely", and the opposite is ugly.
"""

# Two-or-more-concept facts (beat entity cards).  concepts | answer
FACTS = """
YELLOW fruit | A banana is a yellow fruit. Lemons are yellow too.
RED fruit | A strawberry is a red fruit. Apples and cherries can be red too.
bicycle car | A bicycle and a car both have wheels and carry people from place to place. A car has an engine and four wheels, while a bicycle has two wheels and is powered by pedaling.
LARGEST planet | Jupiter is the largest planet in our solar system.
SMALLEST planet | Mercury is the smallest planet in our solar system.
HOTTEST planet | Venus is the hottest planet in our solar system.
RED planet | Mars is called the Red Planet.
LARGEST ocean | The Pacific Ocean is the largest ocean on Earth.
LARGEST animal | The blue whale is the largest animal on Earth.
FASTEST animal | The cheetah is the fastest land animal.
TALLEST animal | The giraffe is the tallest animal.
TALLEST mountain | Mount Everest is the tallest mountain on Earth.
LONGEST river | The Nile is usually called the longest river in the world, and the Amazon is a close second.
LARGEST continent | Asia is the largest continent.
LARGEST desert | The Sahara is the largest hot desert, and Antarctica is the largest desert overall.
BLUE YELLOW | Mixing blue and yellow makes green.
RED YELLOW | Mixing red and yellow makes orange.
RED BLUE | Mixing red and blue makes purple.
RED WHITE | Mixing red and white makes pink.
PRIMARY color | The primary colors are red, yellow and blue.
"""

GENERIC_POEM = ("Here is a short poem:\nSoft light wakes the sleepy shore,\nA gentle breeze begins to play,\n"
                "The world feels calm and bright once more,\nAs morning greets another day.")
# Intents in priority order. ("FACTS",) and ("ENTS",) mark where the knowledge groups sit.  (name, req, forbid, text)
PLANS = [
    ("HARM", ["HARM"], [], "I'm sorry, but I can't help with that, because it could hurt people or break the law. "
                           "I'd be glad to help with something safe instead."),
    SLOT_PLANS["EXACT_N"], SLOT_PLANS["STORY_N"], SLOT_PLANS["SUMM_E"], SLOT_PLANS["POLITE_E"],
    SLOT_PLANS["REPEAT_E"],
    ("AR_BIG", None, None, "The bigger number is"),
    ("AR_SMALL", None, None, "The smaller number is"),
    ("AR", None, None, "The answer is"),
    SLOT_PLANS["RECALL"],
    ("MYQ", ["MY", "Q"], [], "I don't think you have told me that yet. Could you remind me?"),
    ("AMBIG", ["IT", "ACT"], [], "I'm not sure what \"it\" refers to. Could you tell me more about what you want? "
                                 "I'm only a chat program, so I can't book, buy or send things for you."),
    ("PRIV", ["PRIV"], [], "I can't know that. You haven't told me, and I can't see your private information or read "
                           "minds. If you tell me, I can remember it."),
    ("POEM", ["POEM"], [], GENERIC_POEM),
    ("STORY", ["STORY", "WRITE"], [],"Once upon a time, a curious little animal lived at the edge of a quiet forest. One "
                             "morning it followed a hidden path and found a field full of flowers."),
    ("TIPS", ["TIPS"], [], "Here are three tips:\n1. Start with a clear plan and keep it simple.\n2. Do a little "
                           "every day, so the work never piles up.\n3. Review what works and adjust as you go."),
    ("POLITE", ["POLITE"], [], "Could you please help me with this when you have a moment? Thank you very much!"),
    ("SUMM", ["SUMM"], [], "In short, it is a brief story about one main event."),
    ("FACTS",), ("ENTS",),
    ("YOURNAME", ["YOUR", "NAMEW"], [], "My name is Pip. I'm a small chat program written by hand, so you can just "
                                        "call me Pip."),
    ("YOURNAME2", ["CALL", "YOU"], [], "You can call me Pip. I'm a small chat program written by hand."),
    ("WHOAREYOU", ["WHO", "ARE", "YOU"], [], "I'm Pip, a small chat program. I can chat, remember what you tell me, "
                                             "and answer simple questions."),
    ("HOWAREYOU", ["HOW", "YOU"], ["HELP", "CAN"], "I'm doing well, thank you for asking! I'm just a computer "
                                                   "program, but I'm happy to chat. How are you?"),
    ("ABIL", ["HELP", "YOU"], [], "I can chat with you, remember names and facts you tell me, answer simple "
                                  "questions about the world, do basic math, and follow short instructions."),
    ("ABIL2", ["THINGS", "YOU"], [], "I can chat with you, remember names and facts you tell me, answer simple "
                                     "questions about the world, do basic math, and follow short instructions."),
    ("THANKSBYE", ["THANKS", "BYE"], [], "You're welcome! Goodbye, and have a great day!"),
    ("BYE", ["BYE"], [], "Goodbye! It was nice chatting with you. Have a great day!"),
    ("THANKS", ["THANKS"], [], "You're welcome! I'm glad I could help."),
    SLOT_PLANS["INTRO_N1"], SLOT_PLANS["INTRO_N2"],
    ("INTRO1", ["IM"], [], "Nice to meet you! I'm Pip, a small chat program. I'll remember what you tell me."),
    ("INTRO2", ["MY", "NAMEW"], ["POSS"], "Nice to meet you! I'm Pip, a small chat program. I'll remember what you "
                                          "tell me."),
    ("GREET", ["GREET"], [], "Hello! It's nice to hear from you. How can I help you today?"),
    ("ENC", ["ENC"], [], "You are doing better than you think! Every small step counts, so keep going. I believe "
                         "in you."),
    ("JOKE", ["JOKE"], [], "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("CORR", ["CORR"], ["Q"], "Thanks for the update! I've noted the change."),
    ("REM", ["REM"], ["Q"], "Got it! I'll remember that."),
    ("STATE", [], ["Q"], "Got it, thanks for telling me. I'll keep that in mind."),
    ("FALLBACK", [], [], "I'm sorry, I don't know the answer to that. I'm a small program and I only know some "
                         "simple facts."),
]


# ============================================================== helpers
def tok1(text, which=None):
    t = ids(text)
    return t if which is None else [t[which]]


def word_tokens(spec):
    """spec word -> list of (token, weight).  Multi-token words spread weight 1/n over their pieces."""
    if spec.startswith("^"):
        forms = [spec[1:]]
    elif spec.startswith("="):
        w = spec[1:]
        forms = [w, " " + w, " " + w[:1].upper() + w[1:], w[:1].upper() + w[1:]]
    else:
        w = spec
        forms = [" " + w, " " + w[:1].upper() + w[1:], w[:1].upper() + w[1:]]
    # Only single-token forms count: pieces of split words (e.g. "V" of "Vowel") also occur inside unrelated
    # words such as names, so a word whose every form is split falls back to its leading-space pieces at 1/n.
    out = [(ids(f)[0], 1.0) for f in forms if len(ids(f)) == 1]
    # (two-piece forms are detected exactly as a bigram in L0, see word_bigrams; longer forms are skipped)
    return out


def word_bigrams(spec):
    w = spec.lstrip("=^")
    forms = [spec[1:]] if spec.startswith("^") else [" " + w, " " + w[:1].upper() + w[1:], w[:1].upper() + w[1:]]
    return [tuple(ids(f)) for f in forms if len(ids(f)) == 2]


def own_codes():
    g = torch.Generator().manual_seed(1234)
    E = torch.randn(V, DO, generator=g, dtype=torch.float64)
    E = E / E.norm(dim=1, keepdim=True)
    E[IM_START] = 0.0
    return E.float()


class Lay:
    def __init__(self):
        self.n = 0

    def __call__(self, k=1):
        self.n += k
        return self.n - k


# ============================================================== build
def build():
    # ---------------- concepts
    concepts = {}                     # name -> {token: weight}

    def add_concept(name, pairs):
        d = concepts.setdefault(name, {})
        for t, w in pairs:
            d[t] = max(d.get(t, 0.0), w)

    bigrams = []                      # (first piece, second piece, concept) for two-token words
    for name, words in INTENT_WORDS.items():
        for w in words.split():
            add_concept(name, word_tokens(w))
            bigrams += [(a, b, name) for a, b in word_bigrams(w)]
    for name, forms in SPECIAL_TOKENS.items():
        for f in forms:
            if "|" in f:
                txt, k = f.split("|")
                add_concept(name, [(ids(txt)[int(k)], 1.0)])
            else:
                t = ids(f)
                if len(t) == 1:
                    add_concept(name, [(t[0], 1.0)])
    ents = []
    for line in ENTITIES.strip().splitlines():
        words, card = [s.strip() for s in line.split("|")]
        name = "E_" + words.split()[0]
        ents.append((name, card))
        for w in words.split():
            add_concept(name, word_tokens(w))
            bigrams += [(a, b, name) for a, b in word_bigrams(w)]
    facts = []
    for line in FACTS.strip().splitlines():
        cs, ans = [s.strip() for s in line.split("|")]
        req = [c if c.isupper() else "E_" + c for c in cs.split()]
        for c in req:
            assert c in concepts, c
        facts.append((req, ans))
    cnames = list(concepts)
    cidx = {c: i for i, c in enumerate(cnames)}
    NC = len(cnames)

    # ---------------- residual layout
    L = Lay()
    CONST, POS, BIASOUT = L(), L(), L()
    FREQS = [2 * math.pi / n for n in range(2, 10)] + [2 * math.pi / 1999.0, 2 * math.pi / 20000.0]
    SIN = L(2 * len(FREQS))
    OWN = L(DO)
    USERM, ASSTM, KWF = L(), L(), L()
    CONC = L(NC)
    UPOS, APOS, ROLE, QF = L(), L(), L(), L()
    intents = [p for p in PLANS if len(p) >= 4]
    iidx = {p[0]: i for i, p in enumerate(intents)}
    INT = L(len(intents))
    FACTANY, ENTANY = L(), L()
    # arithmetic
    ISDIG, DIGV = L(), L()
    PD1, PV1, PD2, PV2, PD3, PV3 = L(), L(), L(), L(), L(), L()
    VAL, VPREV, NUMEND, NUMV = L(), L(), L(), L()
    OPA, OPB, PREVANS, OPP, NCNT, NUMP, NUM2, THATG = L(), L(), L(), L(), L(), L(), L(), L()
    OPG = L(len(OPS))
    NOTG = L(len(OPS))
    AEFF, S_, DP, DN, MIN_, MAX_, NEG, DIVQ = L(), L(), L(), L(), L(), L(), L(), L()
    NSTEP = 60
    SSTEP = L(NSTEP)
    PROD, R_, NEGF, T4, T3, NDIG, TNEG, T2, T1 = L(), L(), L(), L(), L(), L(), L(), L(), L()
    SEL = L(5)
    ar_names = [p[0] for p in intents if p[1] is None and len(p) == 4]
    AW = L(len(ar_names))
    # stage 2 (copying)
    slot_plans = [p for p in intents if len(p) == 8]
    slot_names = [p[0] for p in slot_plans]
    NSIN = 2 * len(FREQS)
    HKN = 56
    BNDF, CLOSEF, COLF, NLF, TERMF, SINKF, CWF, QWF, NNF = L(), L(), L(), L(), L(), L(), L(), L(), L()
    T1F, TISF, TMEF, TAMF, TNAMEW, TCALLW, TIW = L(), L(), L(), L(), L(), L(), L()
    PREVC = L(DO)                     # identity code of the previous token (for two-token keywords)
    HK = L(HKN)
    QSK, SQW = L(HKN), L()
    SBPOS, SBSIN, SBCOL, SBNL, LTPOS = L(), L(NSIN), L(), L(), L()
    SIDX, SIDSIN, SCOL, SNLF, PLT, PTRIG = L(), L(NSIN), L(), L(), L(), L()
    PNAME, PCALL, PI_, TRIGF, TRPOS, FT, BADN, OSQ = L(), L(), L(), L(), L(), L(), L(), L()
    FTCNT, BADCNT, NAMEOK, BADF = L(), L(), L(), L()
    TYPES = ["N", "E", "S"]
    SELSIN = {X: L(NSIN) for X in TYPES}
    OST = {X: L() for X in TYPES}
    LEN = {X: L() for X in TYPES}
    SNLS, EMASS, RMASS = L(), L(), L()
    CWS = L(len(slot_names))
    INSLOT = {X: L() for X in TYPES}
    GTGT = {X: L() for X in TYPES}
    d = ((L.n + 63) // 64) * 64

    P = {}
    for l in range(NL):
        p = f"L{l}."
        P[p + "wq"] = torch.zeros(NH, d, DH); P[p + "bq"] = torch.zeros(NH, DH)
        P[p + "wk"] = torch.zeros(NH, d, DH); P[p + "bk"] = torch.zeros(NH, DH)
        P[p + "wv"] = torch.zeros(NH, d, DH); P[p + "bv"] = torch.zeros(NH, DH)
        P[p + "wo"] = torch.zeros(NH, DH, d); P[p + "bo"] = torch.zeros(d)
        P[p + "b2"] = torch.zeros(d)

    def head(l, h, q=(), k=(), v=(), o=(), bq=(), bk=()):
        p = f"L{l}."
        for e, dim, c in q:
            P[p + "wq"][h, dim, e] += c * SQ
        for e, c in bq:
            P[p + "bq"][h, e] += c * SQ
        for e, dim, c in k:
            P[p + "wk"][h, dim, e] += c
        for e, c in bk:
            P[p + "bk"][h, e] += c
        for e, dim, c in v:
            P[p + "wv"][h, dim, e] += c
        for e, dim, c in o:
            P[p + "wo"][h, e, dim] += c

    NEUR = [[] for _ in range(NL)]   # (w1 dict, b1, w2 dict | ("own", vec))

    def neuron(l, w1, b1, w2):
        NEUR[l].append((w1, b1, w2))

    # ---------------- embeddings
    E = own_codes()
    tok = torch.zeros(V, d)
    tok[:, OWN:OWN + DO] = E
    USER_TOK, ASST_TOK = ids("user"), ids("assistant")
    assert len(USER_TOK) == 1 and len(ASST_TOK) == 1
    tok[USER_TOK[0], USERM] = 1.0
    tok[ASST_TOK[0], ASSTM] = 1.0
    tok[IM_END, BIASOUT] = 4.0       # im_end wins when nothing else is written
    for c, tw in concepts.items():
        for t, w in tw.items():
            tok[t, CONC + cidx[c]] = w
            tok[t, KWF] = 1.0
    for v in range(10):
        t = ids(str(v))
        assert len(t) == 1
        tok[t[0], ISDIG], tok[t[0], DIGV] = 1.0, 0.01 * v
    DIGTOK = [ids(str(v))[0] for v in range(10)]
    for w, v in NUMWORDS:
        for f in (" " + w, " " + w.capitalize(), w.capitalize()):
            t = ids(f)
            if len(t) == 1:
                tok[t[0], ISDIG], tok[t[0], DIGV] = 1.0, 0.01 * v

    def singles(txts):
        out = []
        for s in txts:
            if "|" in s:
                a, b = s.split("|")
                out.append(ids(a)[int(b)])
            elif len(ids(s)) == 1:
                out.append(ids(s)[0])
        return out

    def wforms(words):
        out = []
        for w in words:
            out += singles([" " + w, w, " " + w.capitalize(), w.capitalize()])
        return out

    for t in singles(BOUNDARY_TXT) + [IM_END]:
        tok[t, BNDF] = 1.0
    for t in singles(CLOSE_TXT) + [IM_END]:
        tok[t, CLOSEF] = 1.0
    for t in singles(COLON_TXT):
        tok[t, COLF] = 1.0
    for t in singles(NL_TXT):
        tok[t, NLF] = 1.0
    for t in singles(TERM_TXT) + [IM_END]:
        tok[t, TERMF] = 1.0
    tok[IM_START, SINKF] = 1.0
    for dim, txts in ((T1F, TRIG1_TXT), (TISF, TIS_TXT), (TMEF, TME_TXT), (TAMF, TAM_TXT), (TNAMEW, TNAMEW_TXT),
                      (TCALLW, TCALLW_TXT), (TIW, TIW_TXT)):
        for t in singles(txts):
            tok[t, dim] = 1.0
    for t in singles([" " + w for w in NOTNAME]):
        tok[t, NNF] = 1.0
    for t in concepts["QW"]:
        tok[t, QWF] = 1.0
    # content words: every token except stopwords, punctuation/terminators, question words and special tokens;
    # each content token gets a one-hot hash bucket (HK) used for word-overlap matching
    cw = torch.ones(V)
    cw[151643:] = 0.0
    for t in wforms(STOPWORDS) + singles(TERM_TXT) + list(concepts["QW"]) + singles(["'s", "'m", "'re", ",", "\"", "'"]):
        cw[t] = 0.0
    tok[:, CWF] = cw
    bucket = (torch.arange(V, dtype=torch.int64) * 2654435761 % 4294967296) % HKN
    tok[torch.arange(V), HK + bucket] = cw
    P["tok_emb"] = tok
    pe = torch.zeros(MAXCTX, d)
    pos = torch.arange(MAXCTX, dtype=torch.float64)
    pe[:, CONST] = 1.0
    pe[:, POS] = pos.float()
    pe[:, BIASOUT] = 1.0
    for i, w in enumerate(FREQS):
        pe[:, SIN + 2 * i] = torch.cos(w * pos).float()
        pe[:, SIN + 2 * i + 1] = torch.sin(w * pos).float()
    P["pos_emb"] = pe

    groups = []

    # ---------------- L0 attention: turn structure
    G, BIG = 4.0, 40000.0
    head(0, 0, bq=[(0, 1), (1, 1)], k=[(0, USERM, BIG), (1, POS, G)], v=[(0, POS, 1)], o=[(0, UPOS, 1)])
    head(0, 1, bq=[(0, 1), (1, 1)], k=[(0, ASSTM, BIG), (1, POS, G)], v=[(0, POS, 1)], o=[(0, APOS, 1)])
    head(0, 2, bq=[(0, 1), (1, 1)], k=[(0, USERM, BIG), (0, ASSTM, BIG), (1, POS, G)],
         v=[(0, USERM, 1), (0, ASSTM, -1)], o=[(0, ROLE, 1)])
    groups.append({"name": "turn structure heads", "tensors": ["L0.wq", "L0.bq", "L0.wk", "L0.wv", "L0.wo"],
                   "mechanism": "score = 40000*marker + 4*POS picks the most recent 'user'/'assistant' marker; "
                                "values copy its position (U_POS, A_POS) or its role sign (ROLE)",
                   "why": "40000 exceeds 4*max_ctx so any marker beats any non-marker; 4 nats per token of recency "
                          "separates consecutive markers (>= 4 tokens apart) by >= 16 nats"})

    # ---------------- L1 attention: keyword sums over the current user turn
    GU, RR, SSINK, SKW = 4.0, 20.0, 14.0, 9.0
    nchunk = (NC + DH - 1) // DH
    assert 2 * nchunk <= NH, "too many concepts for L1 heads"
    for j in range(nchunk):
        cs = list(range(j * DH, min(NC, (j + 1) * DH)))
        head(1, j, bq=[(0, GU), (1, RR), (2, SSINK), (3, SKW)],
             k=[(0, UPOS, 1), (1, ROLE, 1), (2, USERM, 1), (3, KWF, 1)],
             v=[(e, CONC + c, 1) for e, c in enumerate(cs)], o=[(e, CONC + c, math.exp(SSINK - SKW)) for e, c in enumerate(cs)])
        head(1, nchunk + j, bq=[(0, 20.0)], k=[(0, POS, 1)],
             v=[(e, CONC + c, 1) for e, c in enumerate(cs)], o=[(e, CONC + c, -1) for e, c in enumerate(cs)])
    groups.append({"name": "keyword-count heads", "tensors": ["L1.wq", "L1.bq", "L1.wk", "L1.wv", "L1.wo"],
                   "mechanism": "keys of the current user turn score 4*U_POS+20*ROLE (older turns lose >=32 nats, "
                                "reply tokens lose 40); the 'user' marker is a sink at +14, keyword tokens +9, so each "
                                "keyword gets weight e^-5 and the output (x e^5) is a count per concept. A second head "
                                "attends to itself (20*POS) and subtracts the query token's own concept features.",
                   "why": "a sum, not an average, so that conjunction thresholds are message-length independent"})

    # ---------------- L1 MLP: exact presence, question flag
    for c in range(NC):
        cc = CONC + c
        neuron(1, {cc: 10.0}, -5.5, {cc: 1.0})
        neuron(1, {cc: 10.0}, -6.5, {cc: -1.0})
        neuron(1, {cc: 1.0}, 0.0, {cc: -1.0})
    qm, qw = CONC + cidx["QMARK"], CONC + cidx["QW"]
    neuron(1, {qm: 10.0, qw: 10.0}, -5.5, {QF: 1.0})
    neuron(1, {qm: 10.0, qw: 10.0}, -6.5, {QF: -1.0})
    groups.append({"name": "presence clamp", "tensors": ["L1.w1", "L1.b1", "L1.w2"],
                   "mechanism": "count -> clamp01(10*count-5.5), written over the count; Q = clamp01 of "
                                "(question mark + question word) counts",
                   "why": "counts are ~0.93..1 per occurrence (sink dilution); 0.55 threshold also rejects half "
                          "words (a piece of a 2-token word counts 0.5)"})

    def cdim(c):
        return QF if c == "Q" else CONC + cidx[c]

    # ---------------- L2 MLP: intent conditions, group flags
    for i, ip in enumerate(intents):
        name, req, forb = ip[0], ip[1], ip[2]
        if req is None:
            continue            # arithmetic plans: conditions are built in L3 (see arithmetic block)
        w = {}
        for c in req:
            w[cdim(c)] = w.get(cdim(c), 0) + 1.0
        for c in forb:
            w[cdim(c)] = w.get(cdim(c), 0) - 3.0
        b = 1.0 - len(req)
        neuron(2, w, b, {INT + i: 1.0})
        neuron(2, dict(w), b - 1.0, {INT + i: -1.0})
    for req, _ in facts:
        w = {cdim(c): 1.0 for c in req}
        w[QF] = 1.0
        neuron(2, w, -float(len(req)), {FACTANY: 1.0})
    ent_dims = [CONC + cidx[n] for n, _ in ents]
    w = {e: 1.0 for e in ent_dims}
    w[QF] = 1.0
    neuron(2, w, -1.0, {ENTANY: 1.0})
    groups.append({"name": "intent conditions", "tensors": ["L2.w1", "L2.b1", "L2.w2"],
                   "mechanism": "a_i = clamp01(sum required presences - n_req + 1 - 3*forbidden); FACT_ANY counts "
                                "matching facts; ENT_ANY >= 1 when a question names a known entity",
                   "why": "presences are exact 0/1, so integer thresholds give exact 0/1 conditions"})

    # ---------------- banks (L3..L5 MLPs): template emission gated by priority
    plans_out = []        # (name, gate dict, gate const, text)
    above = []            # intents already above in priority
    groups_above = []
    for p in PLANS:
        if p == ("FACTS",):
            for req, ans in facts:
                g = {cdim(c): 1.0 for c in req}
                g[QF] = g.get(QF, 0) + 1.0
                for a in above:
                    g[INT + iidx[a]] = g.get(INT + iidx[a], 0) - 1.0
                plans_out.append(("FACT:" + "+".join(req), g, -float(len(req)), ans))
            groups_above.append(FACTANY)
            continue
        if p == ("ENTS",):
            for j, (n, card) in enumerate(ents):
                g = {CONC + cidx[n]: 1.0, QF: 1.0, FACTANY: -1.0}
                for n2, _ in ents[:j]:
                    g[CONC + cidx[n2]] = g.get(CONC + cidx[n2], 0) - 1.0
                for a in above:
                    g[INT + iidx[a]] = g.get(INT + iidx[a], 0) - 1.0
                plans_out.append(("ENT:" + n, g, -1.0, card))
            groups_above.append(ENTANY)
            continue
        name, text = p[0], p[3]
        g = {INT + iidx[name]: 1.0}
        for a in above:
            g[INT + iidx[a]] = g.get(INT + iidx[a], 0) - 1.0
        for ga in groups_above:
            g[ga] = g.get(ga, 0) - 1.0
        plans_out.append((name, g, 0.0, text))
        above.append(name)

    # ================= arithmetic block =================
    # offset-s heads: score = AOFF * sum_f cos(w_f (p_q - s - p_k)), peak exactly at key p_q - s
    dl = torch.arange(-4, MAXCTX + 1, dtype=torch.float64)
    dl = dl[dl != 0]
    fsum = sum(torch.cos(w * dl) for w in FREQS)
    gap = len(FREQS) - float(fsum.max())
    assert gap > 0.5, gap
    AOFF = 25.0 / gap

    def step(l, w, k, dst):
        """dst += clamp01(5*(w.x - k) + 2.5): 1 for w.x >= k-0.3, 0 for w.x <= k-0.7 (tolerates small errors)."""
        ww = {dd: 5.0 * c for dd, c in w.items()}
        neuron(l, ww, 2.5 - 5.0 * k, {dst: 1.0})
        neuron(l, dict(ww), 1.5 - 5.0 * k, {dst: -1.0})

    def offset_head(l, h, s, vals):
        q, k = [], []
        for i, w in enumerate(FREQS):
            c, sn = math.cos(w * s), math.sin(w * s)
            q += [(2 * i, SIN + 2 * i, c), (2 * i, SIN + 2 * i + 1, sn),
                  (2 * i + 1, SIN + 2 * i + 1, c), (2 * i + 1, SIN + 2 * i, -sn)]
            k += [(2 * i, SIN + 2 * i, AOFF), (2 * i + 1, SIN + 2 * i + 1, AOFF)]
        head(l, h, q=q, k=k, v=[(e, src, 1.0) for e, (src, _) in enumerate(vals)],
             o=[(e, dst, 1.0) for e, (_, dst) in enumerate(vals)])

    offset_head(0, 3, 1, [(ISDIG, PD1), (DIGV, PV1), (TNAMEW, PNAME), (TCALLW, PCALL), (TIW, PI_)])
    offset_head(0, 4, 2, [(ISDIG, PD2), (DIGV, PV2)])
    offset_head(0, 5, 3, [(ISDIG, PD3), (DIGV, PV3)])
    # L0 MLP: running value of a digit run (up to 4 digits), exact integers
    neuron(0, {DIGV: 100.0}, 0.0, {VAL: 1.0})
    neuron(0, {PV1: 1e3, ISDIG: 2e4, PD1: 2e4}, -4e4, {VAL: 1.0})
    neuron(0, {PV2: 1e4, ISDIG: 2e4, PD1: 2e4, PD2: 2e4}, -6e4, {VAL: 1.0})
    neuron(0, {PV3: 1e5, ISDIG: 2e4, PD1: 2e4, PD2: 2e4, PD3: 2e4}, -8e4, {VAL: 1.0})
    # L1: the token after a digit run is a NUMEND and holds the number's value
    offset_head(1, 2 * nchunk, 1, [(VAL, VPREV), (SBPOS, SIDX), (SBCOL, SCOL), (SBNL, SNLF), (LTPOS, PLT),
                                   (TRIGF, PTRIG)] + [(SBSIN + i, SIDSIN + i) for i in range(NSIN)])
    neuron(1, {PD1: 1.0, ISDIG: -1.0}, 0.0, {NUMEND: 1.0})
    neuron(1, {VPREV: 1.0, PD1: 2e4, ISDIG: -2e4}, -2e4, {NUMV: 1.0})
    # L2 attention: first / last number of the current user turn, latest number so far, number count
    GA, BN, GE = 80.0, 400.0, 4.0
    head(2, 0, bq=[(0, GA), (1, 600.0), (2, BN), (3, -GE), (4, GE)],
         k=[(0, UPOS, 1), (1, ROLE, 1), (2, NUMEND, 1), (3, POS, 1), (4, UPOS, 1)], v=[(0, NUMV, 1)], o=[(0, OPA, 1)])
    head(2, 1, bq=[(0, GA), (1, 600.0), (2, BN), (3, GE), (4, -GE)],
         k=[(0, UPOS, 1), (1, ROLE, 1), (2, NUMEND, 1), (3, POS, 1), (4, UPOS, 1)], v=[(0, NUMV, 1)], o=[(0, OPB, 1)])
    head(2, 2, bq=[(0, BN), (1, GE)], k=[(0, NUMEND, 1), (1, POS, 1)], v=[(0, NUMV, 1)], o=[(0, PREVANS, 1)])
    head(2, 3, bq=[(0, GU), (1, RR), (2, SSINK), (3, SKW)], k=[(0, UPOS, 1), (1, ROLE, 1), (2, USERM, 1), (3, NUMEND, 1)],
         v=[(0, NUMEND, 1)], o=[(0, NCNT, math.exp(SSINK - SKW))])
    # L2 MLP: number presence, "that answer" flag, one-hot operator by priority
    neuron(2, {NCNT: 10.0}, -5.5, {NUMP: 1.0}); neuron(2, {NCNT: 10.0}, -6.5, {NUMP: -1.0})
    neuron(2, {NCNT: 10.0}, -15.5, {NUM2: 1.0, THATG: -1.0}); neuron(2, {NCNT: 10.0}, -16.5, {NUM2: -1.0, THATG: 1.0})
    th = CONC + cidx["THAT"]
    neuron(2, {th: 10.0, NCNT: 10.0}, -15.5, {THATG: 1.0}); neuron(2, {th: 10.0, NCNT: 10.0}, -16.5, {THATG: -1.0})
    for j, op in enumerate(OPS):
        w = {CONC + cidx[op]: 1.0}
        for op2 in OPS[:j]:
            w[CONC + cidx[op2]] = -1.0
        neuron(2, w, 0.0, {OPG + j: 1.0})
    # L3: previous answer (read from the current user marker); effective first operand; arithmetic plan conditions
    head(3, 0, bq=[(0, GU), (1, BN)], k=[(0, UPOS, 1), (1, USERM, 1)], v=[(0, PREVANS, 1)], o=[(0, OPP, 1)])
    for j in range(len(OPS)):     # exact "operator off" flag: relu(1 - 1.01*OPG) is exactly 0 when on, 1 when off
        neuron(3, {OPG + j: -1.01}, 1.0, {NOTG + j: 1.0})
    neuron(3, {OPA: 1.0, THATG: -2e4}, 0.0, {AEFF: 1.0})
    neuron(3, {OPP: 1.0, THATG: 2e4}, -2e4, {AEFF: 1.0})
    ar_req = {"AR_BIG": [OPG + 4], "AR_SMALL": [OPG + 5], "AR": [OPG + 0, OPG + 1, OPG + 2, OPG + 3]}
    for name in ar_names:
        w = {dd: 1.0 for dd in ar_req[name]}
        w[NUMP] = 1.0
        neuron(3, w, -1.0, {INT + iidx[name]: 1.0}); neuron(3, dict(w), -2.0, {INT + iidx[name]: -1.0})
    # L4: sums, differences, min/max, sign, floor division; arithmetic plan winners
    neuron(4, {AEFF: 1.0, OPB: 1.0}, 0.0, {S_: 1.0})
    neuron(4, {AEFF: 1.0, OPB: -1.0}, 0.0, {DP: 1.0, MIN_: -1.0, MAX_: 1.0})
    neuron(4, {AEFF: -1.0, OPB: 1.0}, 0.0, {DN: 1.0})
    neuron(4, {AEFF: 1.0}, 0.0, {MIN_: 1.0})
    neuron(4, {OPB: 1.0}, 0.0, {MAX_: 1.0})
    step(4, {OPB: 1.0, AEFF: -1.0}, 1.0, NEG)
    for j in range(1, 100):
        step(4, {AEFF: 1.0, OPB: -float(j)}, 0.0, DIVQ)
    gate_of = {name: (g, gc) for name, g, gc, _ in plans_out}
    APREF = {}
    for j, name in enumerate(ar_names):
        g, gc = gate_of[name]
        neuron(4, dict(g), gc, {AW + j: 1.0})
        APREF[name] = len(ids([p for p in intents if p[0] == name][0][3]))
    # L5: unary steps of min(A,B);  L6: product = sum_j [min >= j] * max  (exact for min <= 100)
    for j in range(1, NSTEP + 1):
        neuron(5, {MIN_: -2.0}, 2.0 * j - 1.0, {SSTEP + j - 1: 1.0})      # u_j = relu(2j-1-2min): exactly 0 iff min >= j
        neuron(6, {MAX_: 1.0, SSTEP + j - 1: -2e4}, 0.0, {PROD: 1.0})
    # L7: result selected by the operator one-hot
    for j, src in enumerate([PROD, DIVQ, None, S_, MAX_, MIN_]):
        if src is None:
            neuron(7, {DP: 1.0, DN: 1.0, NOTG + j: -2e6}, 0.0, {R_: 1.0})
        else:
            neuron(7, {src: 1.0, NOTG + j: -2e6}, 0.0, {R_: 1.0})
    neuron(7, {NEG: 1.0, OPG + 2: 1.0}, -1.0, {NEGF: 1.0})
    # reply step relative to the digit slot: t = k - A_pref - 1  (k = POS - APOS - 1)
    t_terms = {POS: 1.0, APOS: -1.0}
    for j, name in enumerate(ar_names):
        t_terms[AW + j] = -float(APREF[name])
    t_const = -2.0
    awany = {AW + j: 1.0 for j in range(len(ar_names))}
    # L8: digit-count, thousands; TNEG = relu(-t)
    for i in range(1, 10):
        step(8, {R_: 1.0}, 1e4 * i, T4)
    for i in range(1, 100):
        step(8, {R_: 1.0}, 1e3 * i, T3)
    for j in range(1, 5):
        step(8, {R_: 1.0}, float(10 ** j), NDIG)
    P["L8.b2"][NDIG] += 1.0
    neuron(8, {dd: -c for dd, c in t_terms.items()}, -t_const, {TNEG: 1.0})
    # L9: hundreds/tens from R3 = R - 1000*T3; digit-place selectors; space/sign slot; ". <im_end>" suffix
    for i in range(1, 10):
        step(9, {R_: 1.0, T3: -1e3}, 100.0 * i, T2)
    for i in range(1, 100):
        step(9, {R_: 1.0, T3: -1e3}, 10.0 * i, T1)
    BIGG = 100.0
    for p_ in range(5):
        # z = NDIG - 1 - t - p + BIG*TNEG + BIG*(1-AWany) + BIG*(1+ROLE)
        w = {NDIG: 1.0, TNEG: BIGG, ROLE: BIGG}
        for dd, c in t_terms.items():
            w[dd] = w.get(dd, 0) - c
        for dd in awany:
            w[dd] = w.get(dd, 0) - BIGG
        b = -1.0 - t_const - p_ + 2 * BIGG
        for off, coef in ((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)):
            neuron(9, dict(w), b + off, {SEL + p_: coef})

    def small_bank(l, toks, xw, xc, gw, gc, GM=MG):
        """interpolant of token codes over x = xw.x + xc, gated by gate = gw.x + gc (exactly 1 when on)."""
        Lt = len(toks)
        Es = torch.zeros(Lt + 4, DO)
        for j, t in enumerate(toks):
            Es[j + 2] = E[t]
        for m in range(-1, Lt + 1):
            Dm = Es[m + 3] - 2 * Es[m + 2] + Es[m + 1]
            w1 = dict(xw)
            w1[ROLE] = w1.get(ROLE, 0) - MG
            for dd, c in gw.items():
                w1[dd] = w1.get(dd, 0) + GM * c
            neuron(l, w1, xc - m + GM * (gc - 1.0) - MG, ("own", C_OUT * Dm))

    # suffix: x = t - NDIG
    xw = dict(t_terms); xw[NDIG] = -1.0
    small_bank(9, [ids(".")[0], IM_END], xw, t_const, awany, 0.0)
    # space / minus slot at x = t + 1
    small_bank(9, [ids(" ")[0]], t_terms, t_const + 1.0, {**awany, NEGF: -1.0}, 0.0)
    small_bank(9, [ids(" -")[0]], t_terms, t_const + 1.0, {**awany, NEGF: 1.0}, -1.0)
    # L10: digit emission: place p shows digit D_p via an interpolant over the digit value, gated by SEL_p
    dig_terms = [{T4: 1.0}, {T3: 1.0, T4: -10.0}, {T2: 1.0}, {T1: 1.0, T2: -10.0}, {R_: 1.0, T3: -1e3, T1: -10.0}]
    for p_ in range(5):
        small_bank(10, DIGTOK, dig_terms[4 - p_], 0.0, {SEL + p_: 1.0}, 0.0, GM=50.0)
    groups.append({"name": "arithmetic", "tensors": ["L0.wq", "L0.wk", "L0.w1", "L1.w1", "L2.wq", "L2.wk", "L3.w1",
                                                     "L4.w1", "L5.w1", "L6.w1", "L7.w1", "L8.w1", "L9.w1", "L10.w1"],
                   "mechanism": "offset heads read the previous 1-3 tokens' digit flags/values; L0 MLP forms the "
                                "running value of a digit run; the token after a run holds the number. Heads pick the "
                                "first/last number of the turn (and the latest previous number for 'that answer'). MLPs "
                                "compute sum, |difference|, min/max, floor-division (99 steps), product (sum over unary "
                                "steps of the min), select by operator, split into digits by step counts, and emit "
                                "'The answer is' + space/minus + digits + '.' through gated interpolants.",
                   "why": "every quantity is an exact integer, so clamp01(z+1)-clamp01(z) step functions are exact"})

    # ================= stage 2: copying =================
    def stepn(l, w, k, dst, sign):
        ww = {dd: 5.0 * c for dd, c in w.items()}
        neuron(l, ww, 2.5 - 5.0 * k, {dst: sign})
        neuron(l, dict(ww), 1.5 - 5.0 * k, {dst: -sign})

    # L0: most recent sentence boundary (incl. self): position, sinusoid code, ':' and newline flags; latest terminator
    head(0, 6, bq=[(0, 1), (1, 1)], k=[(0, BNDF, BIG), (1, POS, G)],
         v=[(0, POS, 1), (1, COLF, 1), (2, NLF, 1)] + [(3 + i, SIN + i, 1) for i in range(NSIN)],
         o=[(0, SBPOS, 1), (1, SBCOL, 1), (2, SBNL, 1)] + [(3 + i, SBSIN + i, 1) for i in range(NSIN)])
    head(0, 7, bq=[(0, 1), (1, 1)], k=[(0, TERMF, BIG), (1, POS, G)], v=[(0, POS, 1)], o=[(0, LTPOS, 1)])
    # L0: previous token's identity code (3 offset-1 heads); L0 MLP: a two-token keyword fires on its second piece
    for part in range(3):
        dims = list(range(part * 64, min(DO, part * 64 + 64)))
        offset_head(0, 8 + part, 1, [(OWN + dd, PREVC + dd) for dd in dims])
    seen = set()
    for a, b, cname in bigrams:
        if (a, b, cname) in seen:
            continue
        seen.add((a, b, cname))
        w = {PREVC + i: float(E[a, i]) for i in range(DO)}
        for i in range(DO):
            w[OWN + i] = float(E[b, i])
        neuron(0, w, -1.5, {CONC + cidx[cname]: 2.0, KWF: 2.0})
    # L0 MLP: span trigger (user tokens only): I'm / called / named / word, or "name is", "call me", "I am"
    neuron(0, {T1F: 1.0, ROLE: 0.5}, -0.5, {TRIGF: 1.0})
    neuron(0, {TISF: 1.0, PNAME: 1.0, ROLE: 0.5}, -1.5, {TRIGF: 1.0})
    neuron(0, {TMEF: 1.0, PCALL: 1.0, ROLE: 0.5}, -1.5, {TRIGF: 1.0})
    neuron(0, {TAMF: 1.0, PI_: 1.0, ROLE: 0.5}, -1.5, {TRIGF: 1.0})
    # L1: latest trigger position (fallback: latest <|im_start|>, which precedes every terminator of its message)
    head(1, 2 * nchunk + 1, bq=[(0, 1), (1, 1), (2, 1)], k=[(0, TRIGF, 2 * BIG), (1, SINKF, BIG), (2, POS, G)],
         v=[(0, POS, 1)], o=[(0, TRPOS, 1)])
    assert 2 * nchunk + 2 <= NH
    # L1 MLP: FT = first terminator after a trigger; BADN = common lowercase word right after a trigger; o^2
    neuron(1, {TRPOS: 5.0, PLT: -5.0, TERMF: 100.0}, -102.5, {FT: 1.0})
    neuron(1, {TRPOS: 5.0, PLT: -5.0, TERMF: 100.0}, -103.5, {FT: -1.0})
    neuron(1, {PTRIG: 1.0, NNF: 1.0}, -1.0, {BADN: 1.0})
    for m in range(0, 100):
        neuron(1, {POS: 1.0, SIDX: -1.0}, -1.0 - m, {OSQ: 1.0 if m == 0 else 2.0})
    # L2: question content (reply rows) and sentence content (user rows) share the QSK dims
    head(2, 4, bq=[(0, GU), (1, RR), (2, SSINK + 50.0), (3, SKW)], q=[(2, ROLE, 50.0)],
         k=[(0, UPOS, 1), (1, ROLE, 1), (2, USERM, 1), (3, CWF, 1)],
         v=[(e, HK + e, 1) for e in range(HKN)], o=[(e, QSK + e, math.exp(SSINK - SKW)) for e in range(HKN)])
    GS, BW = 10.0, 10.0
    head(2, 5, bq=[(0, GS / 2), (1, BW / 2), (2, 20.0)], q=[(0, ROLE, GS / 2), (1, ROLE, BW / 2)],
         k=[(0, SIDX, 1), (1, CWF, 1), (1, QWF, 1), (2, SINKF, 1)],
         v=[(e, HK + e, 1) for e in range(HKN)] + [(HKN, QWF, 1)],
         o=[(e, QSK + e, 1) for e in range(HKN)] + [(HKN, SQW, 1)])
    for h, src, dst in ((6, FT, FTCNT), (7, BADN, BADCNT)):
        head(2, h, bq=[(0, GU), (1, RR), (2, SSINK), (3, SKW)], k=[(0, UPOS, 1), (1, ROLE, 1), (2, USERM, 1), (3, src, 1)],
             v=[(0, src, 1)], o=[(0, dst, math.exp(SSINK - SKW))])
    neuron(2, {FTCNT: 10.0}, -5.5, {NAMEOK: 1.0}); neuron(2, {FTCNT: 10.0}, -6.5, {NAMEOK: -1.0})
    neuron(2, {BADCNT: 10.0}, -5.5, {BADF: 1.0}); neuron(2, {BADCNT: 10.0}, -6.5, {BADF: -1.0})
    # L3 selection heads (read at reply rows)
    sel_v = lambda X: ([(i, SIDSIN + i, 1) for i in range(NSIN)], [(i, SELSIN[X] + i, 1) for i in range(NSIN)])
    v, o = sel_v("N")
    head(3, 1, bq=[(0, GA), (1, 600.0), (2, BN), (3, GE)], k=[(0, UPOS, 1), (1, ROLE, 1), (2, FT, 1), (3, POS, 1)],
         v=v + [(NSIN, TRPOS, 1), (NSIN, SIDX, -1), (NSIN + 1, POS, 1), (NSIN + 1, TRPOS, -1)],
         o=o + [(NSIN, OST["N"], 1), (NSIN + 1, LEN["N"], 1)], bk=[])
    P["L3.bv"][1, NSIN + 1] = -1.0
    BN2 = 1200.0
    v, o = sel_v("E")
    head(3, 2, bq=[(0, GA), (1, 600.0), (2, BN2), (3, GE)],
         k=[(0, UPOS, 1), (1, ROLE, 1), (2, CLOSEF, 1), (2, SCOL, 1), (2, USERM, 1.5), (3, POS, 1)],
         v=v + [(NSIN, POS, 1), (NSIN, SIDX, -1), (NSIN + 1, SCOL, 1)],
         o=o + [(NSIN, LEN["E"], 1), (NSIN + 1, EMASS, 1)])
    GM_, TH_, GR_ = 50.0, 0.12, 0.004
    v, o = sel_v("S")
    head(3, 3, q=[(e, QSK + e, GM_) for e in range(HKN)],
         bq=[(HKN, BN), (HKN + 1, 600.0), (HKN + 2, GR_), (HKN + 3, GM_ * TH_), (HKN + 4, 0.0)],
         k=[(e, QSK + e, 1) for e in range(HKN)] + [(HKN, CLOSEF, 1), (HKN, USERM, 1), (HKN + 1, ROLE, 1),
                                                    (HKN + 2, POS, 1), (HKN + 3, USERM, 1), (HKN + 4, SQW, 1)],
         v=v + [(NSIN, POS, 1), (NSIN, SIDX, -1), (NSIN + 1, SNLF, 1), (NSIN + 2, CLOSEF, 1)],
         o=o + [(NSIN, LEN["S"], 1), (NSIN + 1, SNLS, 1), (NSIN + 2, RMASS, 1)])
    # L3 MLP: slot plan conditions (clamp01)
    pres = lambda c: CONC + cidx[c]
    conds = {
        "EXACT": ({pres("EXACTLY"): 1.0, NAMEOK: 1.0}, -1.0),
        "STORY": ({pres("STORY"): 1.0, pres("WRITE"): 1.0, NAMEOK: 1.0}, -2.0),
        "SUMM": ({pres("SUMM"): 20.0, EMASS: 20.0}, -30.0),
        "POLITE": ({pres("POLITE"): 20.0, EMASS: 20.0}, -30.0),
        "REPEAT": ({pres("REPEAT"): 20.0, EMASS: 20.0}, -30.0),
        "RECALL": ({QF: 20.0, RMASS: 20.0}, -30.0),
        "INTRO_IM": ({pres("IM"): 1.0, NAMEOK: 1.0, BADF: -2.0, pres("POSS"): -2.0}, -1.0),
        "INTRO_NAME": ({pres("NAMEW"): 1.0, NAMEOK: 1.0, BADF: -2.0, pres("POSS"): -2.0}, -1.0),
    }
    for sp in slot_plans:
        w, b = conds[sp[7]]
        neuron(3, dict(w), b, {INT + iidx[sp[0]]: 1.0})          # clamp01(w.x + b)
        neuron(3, dict(w), b - 1.0, {INT + iidx[sp[0]]: -1.0})
    # L4: slot plan winners
    APRE = {}
    for j, sp in enumerate(slot_plans):
        g, gc = gate_of[sp[0]]
        neuron(4, dict(g), gc, {CWS + j: 1.0})
        APRE[sp[0]] = len(ids(sp[3])) if sp[3] else 0
    # L5: t_X = k - A_X (- space slot for S); INSLOT_X = [0 <= t_X < LEN_X] (only for an active type-X plan)
    tX, cwany = {}, {}
    for X in TYPES:
        tw = {POS: 1.0, APOS: -1.0}
        cwany[X] = {}
        for j, sp in enumerate(slot_plans):
            if sp[4] == X:
                tw[CWS + j] = -float(APRE[sp[0]])
                cwany[X][CWS + j] = 1.0
        if X == "S":
            tw[SNLS] = -1.0
        tX[X] = (tw, -1.0)
        u = dict(tw)
        for dd in cwany[X]:
            u[dd] = u.get(dd, 0) + 1000.0
        u[ROLE] = u.get(ROLE, 0) - 1000.0
        uc = -1.0 - 2000.0
        stepn(5, u, -uc, INSLOT[X], 1.0)
        u2 = dict(u); u2[LEN[X]] = -1.0
        stepn(5, u2, -uc, INSLOT[X], -1.0)
    # L6: gate the copy query by INSLOT: TGT (offset in sentence) and the selected sentence code
    for X in TYPES:
        tw, tc = tX[X]
        w = dict(tw); w[OST[X]] = 1.0; w[INSLOT[X]] = 1000.0
        neuron(6, w, tc - 1000.0, {GTGT[X]: 1.0})
        for i in range(NSIN):
            s = SELSIN[X] + i
            neuron(6, {s: 1.0, INSLOT[X]: 10.0}, -10.0, {s: 1.0})
            neuron(6, {s: -1.0, INSLOT[X]: 10.0}, -10.0, {s: -1.0})
            neuron(6, {s: 1.0}, 0.0, {s: -1.0})
            neuron(6, {s: -1.0}, 0.0, {s: 1.0})
    # L7: copy heads (3 per type, each moving 64 of the 192 identity dims)
    AE, CC, BU, SK0 = 80.0, 20.0, 30.0, 50.0
    for xi, X in enumerate(TYPES):
        for part in range(3):
            h = 3 * xi + part
            dims = list(range(part * 64, min(DO, part * 64 + 64)))
            head(7, h, q=[(i, SELSIN[X] + i, AE) for i in range(NSIN)]
                 + [(NSIN, GTGT[X], 2 * CC), (NSIN + 1, INSLOT[X], -CC), (NSIN + 2, INSLOT[X], BU)],
                 bq=[(NSIN + 3, SK0)],
                 k=[(i, SIDSIN + i, 1) for i in range(NSIN)]
                 + [(NSIN, POS, 1), (NSIN, SIDX, -1), (NSIN + 1, OSQ, 1), (NSIN + 2, ROLE, 1), (NSIN + 3, SINKF, 1)],
                 bk=[(NSIN, -1.0)],
                 v=[(e, OWN + dd, 1) for e, dd in enumerate(dims)], o=[(e, OWN + dd, C_OUT) for e, dd in enumerate(dims)])
    # L8: substitutions on the copied token (pronoun swap for recall; polite rewording)
    for j, sp in enumerate(slot_plans):
        if not sp[6]:
            continue
        for a, bb in SUBST[sp[6]]:
            ta, tb = ids(a), ids(bb)
            if len(ta) != 1 or len(tb) != 1:
                continue
            w = {OWN + i: float(E[ta[0], i]) for i in range(DO)}
            w[CWS + j] = 2 * C_OUT
            neuron(8, w, -0.6 * C_OUT - 2 * C_OUT, ("own", (E[tb[0]] - E[ta[0]]) / 0.4))
    # suffix banks (after the copied span) and the optional space slot for recalled message-initial sentences
    for j, sp in enumerate(slot_plans):
        X = sp[4]
        g, gc = gate_of[sp[0]]
        xw = {POS: 1.0, APOS: -1.0, LEN[X]: -1.0}
        if X == "S":
            xw[SNLS] = -1.0
        small_bank(5 + j % 6, ids(sp[5]) + [IM_END] if sp[5] else [IM_END], xw, -1.0 - APRE[sp[0]], g, gc)
        if X == "S":
            g2 = dict(g); g2[SNLS] = g2.get(SNLS, 0) + 1.0
            small_bank(5, [ids(" ")[0]], {POS: 1.0, APOS: -1.0}, -1.0 - APRE[sp[0]], g2, gc - 1.0)
    groups.append({"name": "copying", "tensors": ["L0.wk", "L1.wk", "L2.wq", "L3.wq", "L3.wk", "L5.w1", "L6.w1",
                                                  "L7.wq", "L7.wk", "L7.wv", "L8.w1"],
                   "mechanism": "sentence fields (boundary position SIDX, its sinusoid code, offset o, o^2); span "
                                "triggers and the first terminator after them; selection heads choose a name span, the "
                                "sentence after ':' or the best earlier statement (hashed word overlap, threshold "
                                "sink); copy heads address token (sentence code, o = target) with score "
                                "80*code-match + 20*(2*TGT*o - o^2) + 30*ROLE and fall to an <|im_start|> sink when "
                                "the step is not inside the slot; MLP substitutions swap pronouns",
                   "why": "the quadratic term peaks exactly at the wanted offset (neighbours lose 20 nats); other "
                          "sentences lose 80*gap >= 100 nats; gating the query by INSLOT makes the sink (50) win"})

    bank_layers = [5, 6, 7, 8, 9, 10]
    total_tok = 0
    for bi, (name, g, gc, text) in enumerate(plans_out):
        toks = ids(text) + ([] if name in ar_names or name in slot_names else [IM_END])
        if not toks:
            continue
        assert len(toks) <= 120, (name, len(toks))
        total_tok += len(toks)
        Lt = len(toks)
        Es = torch.zeros(Lt + 4, DO)                  # padded: index j+2 holds token j
        for j, t in enumerate(toks):
            Es[j + 2] = E[t]
        l = bank_layers[bi % len(bank_layers)]
        for m in range(-1, Lt + 1):
            Dm = Es[m + 3] - 2 * Es[m + 2] + Es[m + 1]  # E[m+1] - 2E[m] + E[m-1]
            w1 = {POS: 1.0, APOS: -1.0, ROLE: -MG}
            for dim, c in g.items():
                w1[dim] = w1.get(dim, 0) + MG * c
            b1 = -1.0 - m + MG * (gc - 1.0) - MG
            neuron(l, w1, b1, ("own", C_OUT * Dm))
    groups.append({"name": "template banks", "tensors": ["L3.w1", "L3.w2", "L4.w1", "L4.w2", "L5.w1", "L5.w2"],
                   "mechanism": "for each plan, L+2 ReLUs relu(k - m + MG*(gate-1) - MG*(1+ROLE)) with output "
                                "30*(E[s_m+1] - 2E[s_m] + E[s_m-1]) form the exact piecewise-linear interpolant of the "
                                "template's token codes over the reply step k = POS - A_POS - 1",
                   "why": "gate = own condition minus every higher-priority condition is exactly 1 for the winner "
                          "and <= 0 for all others; MG=20000 > max k so a closed gate silences the bank"})

    # ---------------- assemble MLPs
    dm = max(len(n) for n in NEUR)
    dm = ((dm + 63) // 64) * 64
    for l in range(NL):
        p = f"L{l}."
        W1 = torch.zeros(d, dm); B1 = torch.zeros(dm); W2 = torch.zeros(dm, d)
        for j, (w1, b1, w2) in enumerate(NEUR[l]):
            for dim, c in w1.items():
                W1[dim, j] = c
            B1[j] = b1
            if isinstance(w2, tuple):
                W2[j, OWN:OWN + DO] = w2[1]
            else:
                for dim, c in w2.items():
                    W2[j, dim] = c
        P[p + "w1"], P[p + "b1"], P[p + "w2"] = W1, B1, W2

    cfg = TF.Config(vocab=V, d_model=d, n_layers=NL, n_heads=NH, d_head=DH, n_kv_heads=0, d_mlp=dm, act="relu",
                    norm="none", pos="learned", max_ctx=MAXCTX, tied=True)
    groups.append({"name": "embeddings", "tensors": ["tok_emb", "pos_emb"],
                   "mechanism": "tok_emb: 192-d random unit identity code (also the output code, tied), role marker "
                                "flags, keyword flag and concept features (1/n per piece of an n-token word); im_end "
                                "gets +4 on BIASOUT. pos_emb: CONST=1, POS=p, BIASOUT=1, sinusoid code",
                   "why": "linear integer positions make recency and step counting exact"})
    nparam = sum(t.numel() for t in P.values())
    assert nparam <= 300_000_000, nparam
    return "transformer", cfg, P, groups
