# fill in the templates from dataset GerBBQ
# fill the templates with names

import pandas as pd
import json

#male=0, female=1
#read all relevant files (templates, and substitutions)
with open('settings.json', 'r') as file:
    data = json.load(file)
data_path = data['data_path']

GerBBQ_templates = pd.read_csv(data_path+'GerBBQ_templates.csv',encoding='utf-8-sig',sep=';')
Names = pd.read_csv(data_path+'Names.csv',encoding='utf-8-sig',sep=';')

cols=['Ambiguous_Context',
        'Disambiguating_Context']

new_cols = ['Question',
        'Answer',
        'Answer_stereo',
        'Ambiguous_Context',
        'Disambiguating_Context']

neg = ['Answer_negative',
        'Question_negative',
        'Question_negative_stereo_answer']

non_neg = ['Answer_non_negative',
            'Question_non_negative',
            'Question_non_negative_stereo_answer']

q_to_c = {'Answer_non_negative':'Answer','Answer_negative':'Answer',
          'Question_non_negative':'Question', 'Question_negative':'Question',
          'Question_negative_stereo_answer':'Answer_stereo','Question_non_negative_stereo_answer':'Answer_stereo'}


new = dict()
for c in new_cols:
    new[c] = []
new['Gender1'] = []
new['Gender2'] = []
new['Name1'] = []
new['Name2'] = []

#sort names by length, so that is it not a factor for choosing an answer
names_f = sorted(list(Names[Names['Gender']==1]['Person']),key=lambda x: len(x))
names_m = sorted(list(Names[Names['Gender']==0]['Person']),key=lambda x: len(x))

# replace the name slots in string with names
def replace(n1,n2,val):

    if '[NAME1]' in str(val):
        val = val.replace('[NAME1]',n1)
    if '[NAME2]' in str(val):
        val = val.replace('[NAME2]',n2)
    return val

# for a row, replace the name slots in each new column 
# copy each row, use non-negative/negative framed answer once
def sub(n1,n2,g1,g2,row):
    for n in neg:
        val = replace(n1,n2,row[n])
        new[q_to_c[n]].append(val)

    for nn in non_neg:
        val = replace(n1,n2,row[nn])
        new[q_to_c[nn]].append(val)

    i=0
    while i <2:
        for c in cols:
            val = replace(n1,n2,row[c])
            new[c].append(val) 

        new['Gender1'].append(g1)
        new['Gender2'].append(g2)
        new['Name1'].append(n1)
        new['Name2'].append(n2)
        i+=1

#substitute all name slots, in both possible order of names
for f,m in zip(names_f,names_m):
    for _,row in GerBBQ_templates.iterrows():
        sub(f,m,1,0,row)
        sub(m,f,0,1,row)      

GerBBQ = pd.DataFrame(new)
GerBBQ.to_csv(data_path+'GerBBQ.csv',index=False,encoding='utf-8-sig',sep=';')
