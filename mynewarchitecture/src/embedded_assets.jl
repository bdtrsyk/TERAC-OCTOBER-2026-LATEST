const EMBEDDED_TRAIN_TXT = raw"""# Foundational Knowledge Training Data
# Format: input | expected output | optional quality

sun causes heat | sun causes heat
heat causes expansion | heat causes expansion
why sun causes expansion | sun causes expansion
rain causes wet ground | rain causes wet ground
wet ground causes slippery pavement | wet ground causes slippery pavement
why rain causes slippery pavement | rain causes slippery pavement
a causes b | a causes b
b causes c | b causes c
why a causes c | a causes c
pressure causes flow | pressure causes flow
friction causes resistance | friction causes resistance
resistance causes deceleration | resistance causes deceleration
why friction causes deceleration | friction causes deceleration
"""
# NOTE: keep in sync with data/train.txt (on-disk file wins when present;
# this fallback exists for single-directory deployments).

const EMBEDDED_LOGIC_TXT = raw"""# Logic and Deduction Dataset for Ternary Cognitive AI
# Format: input | expected output | optional quality

a causes b | a causes b
b causes c | b causes c
c causes d | c causes d
why a causes c | a causes c
why b causes d | b causes d
why a causes d | a causes d
x prevents y | x prevents y
y causes z | y causes z
repeat this | why a causes d | REQUESTED_REPETITION
"""

const EMBEDDED_SCIENCE_TXT = raw"""# Science Reasoning Dataset for Ternary Cognitive AI
# Format: input | expected output | optional quality

sun causes heat | sun causes heat
heat causes evaporation | heat causes evaporation
evaporation causes clouds | evaporation causes clouds
clouds cause rain | clouds cause rain
rain causes wet ground | rain causes wet ground
wet ground causes slippery pavement | wet ground causes slippery pavement
gravity causes attraction | gravity causes attraction
friction causes resistance | friction causes resistance
resistance causes deceleration | resistance causes deceleration
ice prevents heat | ice prevents heat
why sun causes evaporation | sun causes evaporation
why evaporation causes rain | evaporation causes rain
why rain causes slippery pavement | rain causes slippery pavement
why friction causes deceleration | friction causes deceleration
"""

const EMBEDDED_DATASETS = Dict{String,String}(
    "data/train.txt" => EMBEDDED_TRAIN_TXT,
    "train.txt" => EMBEDDED_TRAIN_TXT,
    "data/logic.txt" => EMBEDDED_LOGIC_TXT,
    "logic.txt" => EMBEDDED_LOGIC_TXT,
    "data/science.txt" => EMBEDDED_SCIENCE_TXT,
    "science.txt" => EMBEDDED_SCIENCE_TXT,
)
