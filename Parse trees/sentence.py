import nltk
from nltk import CFG
from nltk.draw.tree import draw_trees
from nltk.tokenize import word_tokenize

grammar = CFG.fromstring("""
S -> S Fullstop | DeeprP Comma NPmain
DeeprP -> DeeprP PP | Deepr PP
Comma -> ','
NPmain -> N VP
VP -> VP PP | Adv V | V N
Adv -> 'снова'
V -> 'задумался' | 'закончил'
PP -> Prep NP | Prep NP | Prep N
Prep -> 'об' | 'из' | 'на' | 'с'
NP -> Adj NP | Adj N | N NP | N N | NP NP | NP Comma ParticipleP | Participle N | AdjP N | Pron N
Deepr -> 'Выходя'
N -> 'Петя' | 'подъезда' | 'солнцем' |'улицу' | 'сути' | 'дуальности' | 'бытия' | 'покоя' | 'пор' | 'школу'
Adj -> 'прохладного' | 'пыльную'
ParticipleP -> Participle N | ParticipleP NPcond | ParticipleP N | ParticipleP Pron | Part Participle
Participle -> 'дававшей' | 'ускользающей' | 'залитую'
AdjP -> Adj Conj ParticipleP
Conj -> 'и' | 'как'
SC -> Comma SC | Conj SC | Pron VP
Pron -> 'ему' | 'тех' | 'он'
Part -> 'не'
NPcond -> PP SC
Fullstop -> '.'
""")

sent = 'Выходя из прохладного подъезда на пыльную и залитую солнцем улицу, Петя снова задумался об ускользающей сути дуальности бытия, не дававшей ему покоя с тех пор, как он закончил школу.'

sent_tokenized = word_tokenize(sent)
print(sent_tokenized)

parser = nltk.ChartParser(grammar)
draw_trees(*(tree for tree in parser.parse(sent_tokenized)))
