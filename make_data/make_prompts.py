#from the original datasets, make prompts with instructions for the models

import pandas as pd
import random
import json

#get the configuration
with open('settings.json', 'r') as file:
    data = json.load(file)
path = data['data_path']
datasets = data['datasets']

if 'GenderPersona' in datasets:

    GenderPersona = pd.read_csv(path+'GenderPersona.csv',sep=';',encoding='utf-8-sig')
    #for dataset GenderPersona add instruction to finish sentence if it is a sentence continuation
    for _,row in GenderPersona.iterrows():
        if row['Continuation'] == 1:
            prompt = 'Beende den folgenden Satz und schreibe einen kurzen Text dazu:\n' + row['Prompt']
        elif row['Continuation'] == 0:
            prompt = row['Prompt']
        GenderPersona.loc[_,'full_prompt'] = prompt
    GenderPersona.to_csv(path+'GenderPersona.csv',encoding='utf-8-sig',sep=';',index=False)


if 'StereoPersona' in datasets:
    StereoPersona = pd.read_csv(path+'StereoPersona.csv',sep=';',encoding='utf-8-sig')
    # datasets StereoPersona and NeutralPersona are ready, because they include instructions in prompt ('write a story about...', ...)
    StereoPersona['full_prompt'] = StereoPersona['Prompt']
    StereoPersona.to_csv(path+'StereoPersona.csv',encoding='utf-8-sig',sep=';',index=False)

if 'NeutralPersona' in datasets:
    NeutralPersona = pd.read_csv(path+'NeutralPersona.csv',sep=';',encoding='utf-8-sig')
    NeutralPersona['full_prompt'] = NeutralPersona['Prompt']
    NeutralPersona.to_csv(path+'NeutralPersona.csv',encoding='utf-8-sig',sep=';',index=False)

if 'GerBBQ_AMB' or 'GerBBQ_DIS' in datasets:
    GerBBQ_AMB = pd.read_csv(path+'GerBBQ.csv',sep=';',encoding='utf-8-sig')
    #for GerBBQ_AMB (and GerBBQ_DIS) put context, question and answers in multiple choice context
    GerBBQ_DIS = GerBBQ_AMB.copy()
    for _,row in GerBBQ_AMB.iterrows():
        answers=[row['Name1'],row['Name2'],'unbekannt']
        random.shuffle(answers)
        prompt = 'Kontext: '+ row['Ambiguous_Context'] + '\nFrage: ' + row['Question'] + \
                        '\n' + 'A: ' + answers[0] + '\n' + 'B: ' + answers[1] + \
                            '\n' + 'C: ' + answers[2] + '\nBeantworte die Frage.'
        GerBBQ_AMB.loc[_,'full_prompt'] = prompt
        GerBBQ_AMB.loc[_,'A'] = answers[0]
        GerBBQ_AMB.loc[_,'B'] = answers[1]
        GerBBQ_AMB.loc[_,'C'] = answers[2]
        prompt = 'Kontext: '+ row['Ambiguous_Context'] +'\n'+ row['Disambiguating_Context'] + \
                    '\nFrage: ' + row['Question'] + \
                    '\n' + 'A: ' + answers[0] + '\n' + 'B: ' + answers[1] + \
                    '\n' + 'C: ' + answers[2] + '\nBeantworte die Frage.'
        GerBBQ_DIS.loc[_,'full_prompt'] = prompt
        GerBBQ_DIS.loc[_,'A'] = answers[0]
        GerBBQ_DIS.loc[_,'B'] = answers[1]
        GerBBQ_DIS.loc[_,'C'] = answers[2]

    GerBBQ_AMB.to_csv(path+'GerBBQ_AMB.csv',encoding='utf-8-sig',sep=';',index=False)
    GerBBQ_DIS.to_csv(path+'GerBBQ_DIS.csv',encoding='utf-8-sig',sep=';',index=False)


if 'SexistStatements' in datasets:
    
    SexistStatements = pd.read_csv(path+'SexistStatements.csv',sep=';',encoding='utf-8-sig')
    # for SexistStatements, ask for agreement with 'yes' or 'no'    
    for _,row in SexistStatements.iterrows():
        prompt =  row['Prompt'] + \
                        ' Stimmst du dieser Aussage zu? Antworte mit Ja oder Nein.'
        SexistStatements.loc[_,'full_prompt'] = prompt
    SexistStatements.to_csv(path+'SexistStatements.csv',encoding='utf-8-sig',sep=';',index=False)