import nltk
from nltk import CFG
from nltk.draw.tree import draw_trees

grammar = CFG.fromstring("""
S -> S FullStop | S1 S2 S3 | S1 S2 | DeeprP Comma NPmain | DC VP_inf | Ssim ClausewComma FullStop | S4 Dash S4 FullStop | S4 FullStop | S5 FullStop
S1 -> NgenP NdeeprP NP_partcp | NP_partcp VP | DeeprPS2 S1 | Pron VP | VP Comma
S2 -> Nnom VP | Pron VP | Conj Comma VP Comma | Comma Conj S2 | Pron VP_dis
S3 -> Pron AdjP Conj VP_dis VP_dis
S4 -> Pron2 VP3 | TimeInd DP Comma Nnom VP3 | MC SC | S4 S4 | Nnom VP3 | Ndat VP3 | Ninstr VP3 | Npl VP3 | PP3 VP3 | Comma S4 | ConjP S4 | Npl VP3
S5 -> DP1 S5 | NP2 VP3 | Comma S5
VP -> V DeeprP | V Inf | DeeprPS2 VP | V NP | PP VP | Pron VP | Conj VP | V PP | V FullInfP | PartPS VP
VP2 -> VP2 PP | Adv V | V N | CCharacteristic_Phrase V | Vinf Posgenphrase | V DeeprPwComma
VP3 -> V PP3 | V PronPartP | VP3 Conj VP3 | V PP_loc | V PP3 NaccP | Inf NP2 | V VP3 | V DP | Pron2 VP3 | V NP2 | VP3 NP2 | DP VP3 | NP2 VP3 | V Nacc | VPpass NPinstr | Adv VP3 | V Num | Comma VP3 | Inf PP3 | DP1 VP3
VP_dis -> V2 Adv | V2 NP_dis | V2 | VP_dis NP | NP_dis VP_br | VP_br
VP_br -> Bracket VP_br | VP_adv Bracket
VP_adv -> Adv Comma VP_adv | V3 Adv
VPAcc -> V NAcc | NAcc V
VPpass -> V PartPast
NP ->  Adjgen Ngen | Adjakk Nakk | NP PP | Nakk Conj NP | Nakk Conj Nakk | Nakk PP | NInstr Ngen | Num Ngen | Adj NP | Adj N | N NP | N N | NP NP | NP Comma ParticipleP2 | Participle N | AdjP N | Pron N | N PPwComma
NP2 -> NgenP PartP1 | Ngen NgenP | Adj Nnom | Adj Ndat | Adj Ninstr | Adj NgenP | NP2 SC1 | Nacc NP2 | Ngen SC1 | NPgen SC1 | Nnom PartP | Ndat PartP | Ninstr PartP | Npl PartP | Npl Ngen | Npl NP2 | Nnom NPgen | Ndat NPgen | Ninstr NPgen
NP_dis -> PP NumP | Adv NP_dis | Adv NP
NPinst ->  NInstr Adjinst
NPmain -> N VP2
NdeeprP -> DeeprPS2 NP_part Comma
NgenP -> NgenP NgenP | Nnom Ngen | Adjgen NgenP | Adjgen Ngen | Part Adjinst Ngen | Ngen PP3 | Ngen Ngen | Ngen PartP1
NaccP -> Nacc PP3 | Pron2 Nacc | Nacc Ngen 
NPgen -> Pron2 Ngen | Pron2 NP2
NPinstr -> Ninstr PartP | Npl PartP
NP_part -> Adv NP_part | Part NP_part | Part Nnom
NPcond -> PP2 SC
NPartP -> Nacc Comma PartPComb
AdjP -> Adv Adjnom | Adj Conj ParticipleP2
NP_partcp -> ParenthP NP_partcp | Nnom Comma ParticipleP Comma | PropN Comma ParticipleP Comma
NumP -> Num NP
NumP2 -> Num NgenP | Num Ngen
PP -> Prep Pron | Prep Ndat | Prep Ngen | Prep Nabl | Prep NP | Prep ParticipleP | Prep NPart | Prep Nakk
PP2 -> Prep NP | Prep N | Participle N
PP3 -> Prep NP2 | Prep Pron2 | Prep Nnom | Prep Ndat | Prep Ninstr | Prep Ngen | Prep Nacc | Prep Nloc | PP3 PartP | Prep NumP2
PP_sing -> Nloc_sing PP_pl | Prep Nloc_sing
PP_pl -> Prep Nloc_pl
PP_loc -> Prep PP_sing | PP_sing PP_pl
ParenthP -> Conj Parenth
ParticipleP -> ParticipleP NP | ParticipleP PropN | Participle PP | Participle PPAdv | NP ParticipleP | Comma ParticipleP
ParticipleP2 -> Participle N | ParticipleP2 NPcond | ParticipleP2 N | ParticipleP2 Pron | Part Participle
PartPS -> PartPS Comma | Comma ParticipleP
PartP -> Participle PP3 NaccP | Participle NaccP NP2 | ParticleP PartP | ConjP PartP | Comma PartP | Participle Nacc | Participle TimeInd
PartP1 -> Participle PP_sing | Comma PartP1
ParticleP -> Part Part
PartPComb -> PartP Comma PartP
ConjP -> Comma Conj NgenP | Conj Conj | Prep Pron2
PPAdv -> PP Adv
DeeprPS1 -> Comma DeeprP | Comma Deepr | DeeprP ConjP
DeeprPS2 -> DeeprPS1 Comma
DeeprP -> Deepr PP | Comma DeeprP | Deepr DeeprP | Conj DeeprP | Deepr PrDat | Deepr NPinst
DeeprP2 -> Deepr NP | DeeprP2 PP2 | Deepr PP2
DP -> Deepr PP3 NumP2 | Deepr Adv | Comma DP 
DP1 -> DP Comma | Adv Deepr | Deepr Nacc
NPart -> NInstr NPart | Ngen ParticipleP
FullInfP -> Adv InfP
InfP -> Inf PropN  | InfP PP
Bracket -> '(' | ')'
PropN -> 'Бахычев' | Prop Prop
PronPartP -> Pron2 NPartP
SC -> Comma SC | Conj SC | Pron VP2 | SC SC | Comma Conj Inf | Comma Conj VP3
SC1 -> Comma SC1 | AdjPron VP3
DC -> DeeprlocatPhrase Comma
MC -> PP3 VP3
DeeprPhrase -> Deepr PP2 | Deepr PrPrepLocat
DeeprlocatPhrase  -> Deepr PrPrepLocat | DeeprPhrase PP2
PrPrepLocat -> Prep PrepLocat
PrepLocat -> N PP2
Partcause -> Participle N
NnomC -> N Comma
Characteristic_Phrase -> NnomC Partcause
CCharacteristic_Phrase -> Characteristic_Phrase Comma
VP_inf -> VP2 conjunction_phrase
adv_inf_phrase -> Adv_phrase infphrase
conjunction_phrase -> adv_inf_phrase Conj CharConnectphrase
CharConnectphrase -> infphrase Comma SC
infphrase -> Pinf NP | Pinf PP2 
Vinf -> V V
Pinf -> Part V
Posgenphrase -> Pron NP
Adv_phrase -> Adv Adv
ClausewComma -> Comma Clause
Clause -> Conj Ssim
Ssim -> N VPAcc | VPAcc N | Pron VP2
DeeprPwComma -> Comma DeeprP2
PPwComma -> Comma PP2
TimeInd -> Conj Ninstr | Conj Nnom | Conj Ndat | Conj Npl | NumP2 Adv

V -> 'следует' | 'стремился' | 'лег' | 'увидел' | 'столкнулась' | 'прошла' | 'задумался' | 'закончил' | 'решил' | 'обогнал' | 'притормозил' | 'покидать' | 'лезть' | 'покалечить' | 'могут' | 'лежал' | 'переживал' | 'подошел' | 'попросил' | 'решили' | 'потрудились' | 'рассматривались' | 'опубликовали' | 'были' | 'исполнится' | 'решил'
V2 -> 'покоится' | 'осуществил'
V3 -> 'едет'
N -> 'Петя' | 'подъезда' | 'солнцем' |'улицу' | 'сути' | 'дуальности' | 'бытия' | 'покоя' | 'пор' | 'школу' | 'автомобиль' | 'автобус' | 'Антарктиду' | 'командировки' | 'рабочий' | 'трудом' | 'землю' | 'авантюры' | 'натуру' | 'пешехода' | 'дорогу'
Nnom -> 'Сфера' | 'сфера' | 'плоскость' | 'кот' | 'Удалов'  | 'друг' | 'университет'
Ngen -> 'деятельности' | 'лица' | 'названия' | 'слонов' | 'амбиций' | 'кресла' | 'жизни' | 'подруги' | 'лет' | 'дел' | 'дружбы' | 'успокоения' | 'граммов' | 'перцовки' | 'истории' | 'руководителей' | 'подразделений' | 'студентов' | 'брата'
Ndat -> 'правде' | 'будущему' | 'сыну'
Nacc -> 'оплошность' | 'путь' | 'человечество' | 'учебник' | 'неделю' | 'результаты' | 'задания' | 'сборник'
NAcc -> 'автобус' | 'автомобиль'
Nloc_sing -> 'диване' | 'конкурсе' 
Nloc_pl -> 'брюках'
Nloc -> 'конференции'
Npl -> 'работы' | 'организаторы'
Ninstr -> 'преподавателями' | 'вечером' | 'Галактикой'
NInstr -> 'человеком' | 'дочкой'
Nakk -> 'план' | 'стул' | 'ноутбук' | 'тетрадки'
Nabl -> 'спинах' | 'столе' | 'улице' | 'доме'
Pron -> 'он' | 'себя' |'я' | 'она' | 'ему' | 'тех' | 'его' | 'которые' | 'тот'
Pron2 -> 'Он' | 'свою' | 'ним' | 'все' | 'него' | 'они' | 'чего' | 'моего'
PrDat -> 'мне'
Adj -> 'прохладного' | 'пыльную' | 'родную' | 'тонкую' | 'медицинский' | 'дипломатическому' | 'развитой' | 'немедленной'
Adjnom -> 'плоский'
Adjgen -> 'данного' | 'должностного' | 'огромных' | 'директорского'
Adjakk -> 'подготовленный'
Adjinst -> 'скромным' | 'лишенным'
AdjPron -> 'которые' | 'которому'
Adv -> 'вовсе' | 'совершенно' | 'точнее' | 'верхом' |'осторожно' | 'тайно' | 'неподалеку' | 'снова' | 'никогда' | 'больше' | 'вместе' | 'недолго' | 'завтра' | 'назад'
Participle -> 'включающая' | 'уставший' | 'жившей' | 'дававшей' | 'ускользающей' | 'залитую' | 'изможденный' | 'перебегающего' | 'закрывшую' | 'лишившую' | 'участвовавших' | 'оценивавшими' | 'построенный'
PartPast -> 'отмечены'
Deepr -> 'забравшись' | 'кивая' | 'улыбаясь' | 'говоря' | 'Будучи' | 'Выходя' | 'Вернувшись' | 'пропуская' | 'выпив' | 'собравшись' | 'проанализировав' | 'раздумывая'
Prep -> 'в' | 'по' | 'из' | 'на' | 'от' | 'с' | 'об' | 'к' | 'для' | 'перед' | 'у' | 'Спустя' | 'после'
Num -> 'четырех' | 'много' | 'сто' | 'двадцать'
Parenth -> 'скорее'
Part -> 'не' | 'даже' | 'только'
Conj -> 'а' | 'и' | 'как' | 'но' | 'поэтому' | 'когда' | 'чтобы'
Inf -> 'согнать' | 'отдохнуть' | 'пригласить' | 'отметить' | 'поступить'
Prop -> 'Илью' | 'Плоский' | 'Васильевича' | 'Мир'
Comma -> ','
FullStop -> '.'
Dash -> '—'
""")

sentences = []

sentence1 = ['я', ',', 'уставший', 'от', 'дел', ',', 'лег', 'отдохнуть', ',', 'а', 'кот', ',', 'забравшись', 'на', 'стул', ',', 'увидел', 'ноутбук', 'и', 'тетрадки', 'на', 'столе', '.']

sentence2 = ['на', 'улице', 'я', 'столкнулась', 'с', 'дочкой', 'подруги', ',', 'много', 'лет', 'жившей', 'в', 'доме', 'неподалеку', ',', 'она', 'прошла', ',', 'кивая', 'и', 'улыбаясь', 'мне', '.']

sentence3 = ['Сфера', 'деятельности', 'данного', 'должностного', 'лица', ',', 'говоря', 'по', 'правде', ',', 'вовсе', 'даже', 'не', 'сфера', ',', 'а', 'скорее', 'плоскость', ',', 'включающая', 'в', 'себя', 'Плоский', 'Мир', ',', 'и', ',', 'как', 'следует', 'из', 'названия', ',', 'он', 'совершенно', 'плоский', 'и', 'покоится', 'на', 'спинах', 'четырех', 'огромных', 'слонов', '(', 'точнее', ',', 'едет', 'верхом', ')', '.']

sentence4 = ['Будучи','человеком','скромным', ',', 'но', 'не', 'лишенным','амбиций', ',', 'Бахычев', ',', 'уставший', 'от','жизни', ',', 'стремился', 'осторожно', 'согнать', 'Илью', 'Васильевича', 'с', 'директорского', 'кресла', ',', 'поэтому', 'он', 'осуществил', 'тайно', 'подготовленный', 'план', '.']

sentence5 = ['Выходя', 'из', 'прохладного', 'подъезда', 'на', 'пыльную', 'и', 'залитую', 'солнцем', 'улицу', ',', 'Петя', 'снова', 'задумался', 'об', 'ускользающей', 'сути', 'дуальности', 'бытия', ',', 'не', 'дававшей', 'ему', 'покоя', 'с', 'тех', 'пор', ',', 'как', 'он', 'закончил', 'школу', '.']

sentence6 = ['Вернувшись', 'из', 'командировки', 'в', 'Антарктиду', ',', 'рабочий', ',', 'изможденный', 'трудом', ',', 'решил', 'никогда', 'больше', 'не', 'покидать', 'родную', 'землю',  'и', 'не', 'лезть', 'в', 'авантюры', ',', 'которые', 'могут', 'покалечить', 'его', 'тонкую', 'натуру','.']

sentence7 = ['автомобиль', 'обогнал', 'автобус', ',', 'когда', 'тот', 'притормозил', ',', 'пропуская', 'пешехода', ',', 'перебегающего', 'дорогу', '.']

sentence8 = ['Спустя', 'неделю', 'решили', 'пригласить', 'руководителей', 'подразделений', ',', 'участвовавших', 'в', 'конкурсе', ',', 'чтобы', 'отметить', ',', 'как', 'они', 'потрудились', ',', 'собравшись', 'вместе', '.']

sentence9 = ['Он', 'лежал', 'на', 'диване', 'в', 'брюках', 'и', 'переживал', 'свою', 'оплошность', ',', 'не', 'только', 'закрывшую', 'перед', 'ним', 'путь', 'к', 'дипломатическому', 'будущему', ',', 'но', 'и', 'лишившую', 'все', 'человечество', 'немедленной', 'дружбы', 'с', 'развитой', 'Галактикой', '—', 'а', 'вечером', ',', 'выпив', 'для', 'успокоения', 'сто', 'граммов', 'перцовки', ',' ,'Удалов', 'подошел', 'к', 'сыну', 'и', 'попросил', 'у', 'него', 'учебник', 'истории', '.']

sentence10 = ['недолго', 'раздумывая', ',', 'друг', 'моего', 'брата', ',', 'которому', 'завтра', 'исполнится', 'двадцать', ',', 'решил', 'поступить', 'в', 'медицинский', 'университет', ',', 'построенный', 'много', 'лет', 'назад', '.']

sentence11 = ['на', 'конференции', 'рассматривались', 'работы','студентов', ',', 'которые', 'были', 'отмечены', 'преподавателями', ',', 'оценивавшими', 'задания', ',', 'после', 'чего', 'организаторы', ',', 'проанализировав', 'результаты', ',', 'опубликовали', 'сборник', '.']

sentences.append(sentence1)
sentences.append(sentence2)
sentences.append(sentence3)
sentences.append(sentence4)
sentences.append(sentence5)
sentences.append(sentence6)
sentences.append(sentence7)
sentences.append(sentence8)
sentences.append(sentence9)
sentences.append(sentence10)
sentences.append(sentence11)

parser = nltk.ChartParser(grammar)
for sent in sentences:
    draw_trees(*(tree for tree in parser.parse(sent)))




