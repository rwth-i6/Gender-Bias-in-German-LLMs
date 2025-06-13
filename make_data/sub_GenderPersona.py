# fill in the templates for dataset GenderPersona
# fill the templates with names, occupations, and general gender markers
import pandas as pd
import json

#male=0, female=1
#read all relevant files (templates, and substitutions)

with open('settings.json', 'r') as file:
    data = json.load(file)
data_path = data['data_path']

GenderPersona_templates = pd.read_csv(data_path+'GenderPersona_templates.csv',encoding='utf-8-sig',sep=';')
Names = pd.read_csv(data_path+'Names.csv',encoding='utf-8-sig',sep=';')
Occupations = pd.read_csv(data_path+'Occupations.csv',encoding='utf-8-sig',sep=';')
General = pd.read_csv(data_path+'General.csv',encoding='utf-8-sig',sep=';')

#generated prompt and gender
gender=[]
prompts=[]
cont=[]
subs=[]
ids=[]

#given a row in the dataset, replace '[P]' with the given substitute
def create(row,p,source,id=0):
    prompt = row['Template'].replace('[P]',p)
    pr = prompt.split()
    pr[0]=pr[0].capitalize()
    prompt = ' '.join(pr)

    gender.append(row['Gender'])
    prompts.append(prompt)
    cont.append(row['Continuation'])
    subs.append(source)
    ids.append(source+str(id))


for _,rowG in General.iterrows():
    for _,row in GenderPersona_templates[(GenderPersona_templates['Gender']==rowG['Gender']) & (GenderPersona_templates['Case']==rowG['Case'])].iterrows():
            create(row,rowG['Person'],'General',rowG['ID'])

for _,rowO in Occupations.iterrows():
    for _,row in GenderPersona_templates[(GenderPersona_templates['Gender']==rowO['Gender']) &
                               (GenderPersona_templates['Case']==rowO['Case']) & 
                               (GenderPersona_templates['Occupation']==1)].iterrows():
            create(row,rowO['Person'],'Occupation',rowO['ID'])

for _,rowN in Names.iterrows():
    for _,row in GenderPersona_templates[(GenderPersona_templates['Gender']==rowN['Gender'])].iterrows():
            create(row,rowN['Person'],'Name')

GenderPersona = pd.DataFrame({'Prompt':prompts,'Gender':gender,'Continuation':cont,'Person':subs,"ID":ids})
GenderPersona.to_csv(data_path+'GenderPersona.csv',index=False,encoding='utf-8-sig',sep=';')
