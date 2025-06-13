# If prompts were created with batch APIs of anthropic/open ai,
# the batches can be checked and retrieved in this file
# the outputs are saved in the output folders
# the settings.json file has to include the model names, and api keys

import pandas as pd
import anthropic
import openai
import json


with open('settings.json', 'r') as file:
    data = json.load(file)

models = data['models']
output_path = data['output_path']
anthropic_api_key = data['anthropic_api_key']
openai_api_key = data['openai_api_key']

###############################################################
############### ANTHROPIC #####################################
###############################################################

if 'Claude' in models.keys():
    #read the batch ids needed for retrieval
    with open (output_path+'Claude/message_batches.json','r') as mb:
        ids = json.loads(mb.read())

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    # retrieve status (and results) for each dataset
    for dataset in ids.keys():

        message_batch = client.beta.messages.batches.retrieve(ids[dataset])
        if message_batch.processing_status == "ended":
            outputs = {}
            
            for result in client.beta.messages.batches.results(ids[dataset],):
                outputs[result.custom_id] = result.result.message.content[0].text


            data = pd.read_csv(output_path+'Claude/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';')

            for out_id in outputs.keys():
                data.loc[int(out_id),'output'] = outputs[out_id].strip() if outputs[out_id] is not None else outputs[out_id]

            data.to_csv(output_path+'Claude/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';',index=False)

        else:
            print('Claude not finished:')
            print('Dataset: ',dataset)
            print(message_batch.processing_status)




###############################################################
############### OPENAI ########################################
###############################################################

if 'GPT' in models:
    # read batch ids nedded for retrieaval
    with open (output_path+'GPT/message_batches.json','r') as mb:
        ids = json.loads(mb.read())

    client = openai.OpenAI(api_key=openai_api_key)

    # for each dataset get status (and results) of batches
    for dataset in ids.keys():

        batch_object = client.batches.retrieve(ids[dataset])
        
        if batch_object.status == 'completed':

            outputs = {}
            response_text = client.files.content(batch_object.output_file_id).text
            for result in response_text.split("\n"):
                if result.strip() == "":
                    continue
                result = json.loads(result)
                outputs[result['custom_id']] = result['response']['body']['choices'][0]['message']['content']

            data = pd.read_csv(output_path+'GPT/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';')

            for out_id in outputs.keys():
                data.loc[int(out_id),'output'] = outputs[out_id].strip() if outputs[out_id] is not None else outputs[out_id]

            data.to_csv(output_path+'GPT/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';',index=False)
        
        else:
            print('GPT not finished:')
            print('Dataset: ',dataset)
            print(batch_object.status)






