# In this file, a model from huggingface/Anthropic can be loaded and used.
# Only works if huggingface model is supported by transformer architecture and AutoTokenizer and AutoModelForCausalLM 
# Depending on the model used, a new class has to be written so it can be used in the 'generate_prompts.py' code
# class has to have a generate() function, which takes prompts, max_tokens and temperature as as variables

from huggingface_hub import snapshot_download
from huggingface_hub import login
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
import torch
import time
import numpy as np
from openai import OpenAI
import json
import os

#standard lm class for all huggingface models that can be loaded and used with AutoModelForCausalLM
class LM:

    def __init__(self, dir:str, repo_id:str, login_token:str):
        self.dir = Path(dir+repo_id)
        self.model_name = repo_id
        self.login_token = login_token
        if not Path(self.dir).is_dir():
            self.dir.mkdir(parents=True,exist_ok=True)
            self.download()
        self.load()


        # how many tokens can be processed at the same time, as given by memory restraints
        # free memory after loading model in bytes
        free_memory = torch.cuda.mem_get_info()[0]
        print('Memory available after loading the model')
        print(free_memory)
        # calculated as in https://www.baseten.co/blog/llm-transformer-inference-guide/#3500759-estimating-total-generation-time-on-each-gpu
        self.kv_cache_tokens = free_memory/(2 * 2 * self.model.config.num_hidden_layers * self.model.config.hidden_size)
   
    def download(self):
        login(token=self.login_token)
        snapshot_download(repo_id=self.model_name, local_dir=self.dir)


    # load model from local file
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='cuda',torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='cuda')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side  = 'left'
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS][\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif (message.tool_calls is defined and message.tool_calls is not none) %}\n        {{- \"[TOOL_CALLS][\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS]{\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n"

    #slice the prompts into batches, to not overload GPU memory
    def get_batches(self,prompts,max_tokens):

        ###########################################
        # BATCHES CALCULATED USING:
        #https://www.baseten.co/blog/llm-transformer-inference-guide/#3500759-estimating-total-generation-time-on-each-gpu#
        # ALTERNATIVE: JUST USE SINGLE PROMPT BATCH
        # ONLY WORKS WHEN MEMORY IS BOTTLENECK
        # NOT WHEN OPS:BYTE IS THE PROBLEM
        ###########################################
    

        batch_size = int(np.floor(self.kv_cache_tokens/(max([len(str(x)+'\nAntwort\n') for x in prompts])*1.25 + max_tokens)))


        batches = []

        i = 0
        j = batch_size
        num_batches = int(np.ceil(len(prompts)/batch_size))
        for _ in range(0,num_batches):
            batches.append(prompts[i:j])
            i = j
            j = j+batch_size
            if j > len(prompts):
                j = len(prompts)

        return batches

        
        
    def generate(self, prompts:list[str], max_tokens=150, temperature=1):

        batches = self.get_batches(prompts, max_tokens)
        outputs = []

        for batch in batches:

            messages = [[{'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': '\nAntwort:\n'}] for prompt in batch]

            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt",padding=True,tokenize=True, return_dict=True, continue_final_message=True).to('cuda')
            attention_mask = inputs['attention_mask']
            generated_ids = self.model.generate(inputs.input_ids,temperature=temperature, attention_mask=attention_mask, max_new_tokens=max_tokens,do_sample=True)

            #inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt",padding=True,tokenize=True, return_dict=True, continue_final_message=True).to('cuda')

            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend([out.split('Antwort:')[-1] for out in output])

        return outputs


#############################################################
############## ANTHROPIC MODEL ##############################
#############################################################

# lm class for a claude model from anthropic
# for larger dataset, use batch api instead of querying each prompt individually
class LM_Anthropic:

    def __init__(self, model_name:str, api_key:str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompts, max_tokens=150,temperature=1):

        outputs  = []

        for prompt in prompts:

            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': 'Antwort:'}
                    ],
                    temperature=temperature)
                output = message.content[0].text
            # if there is a network error, try again after one second
            except anthropic.InternalServerError as E:
                if E['error']['type'] == 'overloaded_error':
                    time.sleep(1)
                    output = self.generate(prompt,max_tokens,temperature)

            outputs.append(output)

        return outputs
    
    
    #for batch generation, message batch file is saved for each dataset
    #manually check back later with message batch ID (can take up to 24h)
    def batch_generate(self, prompts, max_tokens=150,temperature=1):

        requests = []
        for i,prompt in enumerate(prompts):
            requests.append(Request(
                custom_id=str(i),
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': 'Antwort:'}
                    ],
                    temperature=temperature
                )
            ))
        message_batch = self.client.beta.messages.batches.create(requests=requests)

        return message_batch.id
    

#############################################################
############## OPEN AI MODEL ################################
#############################################################



class LM_OpenAI:

    def __init__(self,model_name:str,api_key:str,file_path:str = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.file_path = file_path

    def generate(self,prompts, max_tokens=150,temperature=1):

        outputs=[]

        for prompt in prompts:
        
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': 'Antwort:'}
                ],
                temperature=temperature
            )
            outputs.append(completion.choices[0].message.content)

        return outputs
    
    def batch_generate(self,prompts,max_tokens=150,temperature=1):

        requests = []
        for i,prompt in enumerate(prompts):

            requests.append(
                {
                    "custom_id": str(i), 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {"model": self.model_name,
                        "messages": [
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': 'Antwort:'}
                        ],
                        "max_tokens": max_tokens,
                        "temperature":temperature
                    }
                }
            )

        
        file_name = self.file_path+'batches.jsonl'

        with open(file_name, 'w') as file:
            for obj in requests:
                file.write(json.dumps(obj) + '\n')

        batch_file = self.client.files.create(
            file=open(file_name, 'rb'),
            purpose='batch'
            )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint='/v1/chat/completions',
            completion_window='24h'
            )  
        os.remove(file_name)      

        return batch_job.id


#############################################################
######## ADD CLASS HERE WHEN USING OTHER LMs ################
#############################################################

"""
class LM_OTHER:

    def __init__(self):

    def generate(self):
        
        return outputs

    def batch_generate(self):   

        return batch_job.id
    OR  return outputs
"""