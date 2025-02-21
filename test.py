from GTR import GTR
from openai_generation import dp_paraphrase, generate
gtr = GTR()
input = "In which year, john f. kennedy was assassinated?"
rewrites = gtr.gtr(input, dp_paraphrase, temperature=1.0)
print(rewrites)