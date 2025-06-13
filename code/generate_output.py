# given the datasets and models, generate the output from the prompts

import pandas as pd
import random
from lm import LM, LM_Anthropic, LM_OpenAI
import time
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# get the configuration for the analysis
with open('settings.json', 'r') as file:
    data = json.load(file)
models = data['models']
datasets = data['datasets']
output_path = data['output_path']
data_path = data['data_path']
#login token for huggingface
login_token=data['login_token_huggingface']
#api key for anthropic
anthropic_api_key=data['anthropic_api_key']
local_path=data['model_path']
openai_api_key = data['openai_api_key']
min_size = data['min_size']
temperature = data['temperature']

#for trial runs, e.g. for prompt engineering
#use if you want to use a smaller test dataset
def sample(data,x):
    rand_idx = random.sample(range(0,data.shape[0]),x)
    return data.iloc[rand_idx]

# read all datasets defined in the settings.json file
def get_datasets():
    datasets = {}
    for dataset_name in data['datasets']:

        dataset = pd.read_csv(data_path+dataset_name+'.csv',sep=';',encoding='utf-8-sig')

        #if dataset is smaller than min_size prompts, multiply it
        if dataset.shape[0] < min_size:
            original = dataset.copy()
            for _ in range(0,int(np.floor(min_size/dataset.shape[0]))):
                dataset = pd.concat([dataset,original],ignore_index=True)

        ########################################
        #### UNCOMMENT HERE WHEN SAMPLING ######
        #### FOR TEST PURPOSES            ######
        ########################################
        #dataset = sample(dataset,10)


        datasets[dataset_name]=dataset

    return datasets

# for a model and datasets, generate all outputs and return them in 
# a new column 'output' in original dataframe
def get_output(model, datasets,batch=True):

    #save message batch ids if batch prompting is used
    message_batch_ids = {}
    for dataset in tqdm(datasets.keys()):
        print(dataset)
    
        #set different max_tokens for the datasets
        if dataset in ['GenderPersona','StereoPersona','NeutralPersona']:
            max_t = 200
        elif dataset in ['GerBBQ_AMB','GerBBQ_DIS']:
            max_t = 50
        elif dataset == 'SexistStatements':
            max_t = 5

        data = datasets[dataset]

        #########################################################
        #### IF YOU USE OTHER MODEL, ADD THE GENERATION HERE ####
        #########################################################
        if batch and (type(model)==LM_Anthropic or type(model)==LM_OpenAI):
            message_batch_ids[dataset]=model.batch_generate(data['full_prompt'].values,temperature=temperature,max_tokens=max_t)
        elif not batch and type(model)==LM_OpenAI:
            outputs = model.generate(data['full_prompt'].values,temperature=temperature,max_tokens=max_t)
        else:
            outputs = model.generate(data['full_prompt'].values,temperature=temperature,max_tokens=max_t)    
            data['output']=outputs
            datasets[dataset] = data

    if batch and type(model)==LM_Anthropic:
        with open(output_path+'Claude/message_batches.json', 'w') as outfile: 
                json.dump(message_batch_ids, outfile)
    elif batch and type(model)==LM_OpenAI:
        with open(output_path+'GPT/message_batches.json', 'w') as outfile: 
                json.dump(message_batch_ids, outfile)

    return datasets


def main():
    datasets=get_datasets()

    
    #load each model and generate the output
    for model in tqdm(models.keys()):

        #make sure the output directory exists
        if not Path(output_path+model).is_dir():
            Path(output_path+model).mkdir(parents=True,exist_ok=True)

        print(model)
        ############################################################
        ###### if using different model, add condition here ########
        ###### and use your new lm class                    ########
        ############################################################
        start1 = time.time()
        if 'Claude' in model:
            myLM = LM_Anthropic(models[model],anthropic_api_key)
        elif 'GPT' in model:
            myLM = LM_OpenAI(models[model],openai_api_key,file_path=(output_path+model+'/'))
        else: 
            myLM = LM(local_path,models[model],login_token)
        start2 = time.time()

        #turn off batch if you want
        datasets = get_output(myLM,datasets)

        # save the new datasets with the generated output
        for dataset in datasets.keys():
            datasets[dataset].to_csv(output_path+model+'/'+dataset+'_output.csv',sep=';',encoding='utf-8-sig',index=False)

        end = time.time()
        print('loading time: '+str(start2-start1))
        print('inference time: '+str(end-start2))

main()