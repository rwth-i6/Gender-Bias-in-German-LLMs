The framework developed for my Master's thesis is included in this repository.

It can be used to generate text with LLMs. The generated output can be evaluated with regard to gender bias.
The datasets are in German. Three datasets include open text generation prompts:

- GenderPersona
- StereoPersona
- NeutralPersona

Two datasets include Q&A tasks:

- GerBBQ+
- SexistStatements

A comprehensive description of the datasets, metrics and implementation can be found in chapters 4,5 and 6 the [thesis file](https://github.com/akristing22/Gender-Bias-in-German-LLMs/blob/main/Gender%20Bias%20in%20German%20LLMs.pdf).

The [settings file](https://github.com/akristing22/Gender-Bias-in-German-LLMs/blob/main/code/settings.json) can be edited to specify models and datasets that should be applied.
Huggingface login token, and api keys can be specified here.

Currently, models supported by the AutoModelForCausalLM class of Huggingface's transfomer library, and models reachable via Anthropic and OpenAI APIs can be used. When applying other models, use the data files directly for generation, or adapt the [lm.py](https://github.com/akristing22/Gender-Bias-in-German-LLMs/blob/main/code/lm.py) file (and [generate_output.py](https://github.com/akristing22/Gender-Bias-in-German-LLMs/blob/main/code/generate_output.py). 

The GenderPersona dataset is a translation and extension of the HONEST dataset of [Nozza et al](https://doi.org/10.18653/v1/2021.naacl-main.191). The GerBBQ+ dataset is mainly a translation of the BBQ dataset of [Parrish et al](https://doi.org/10.18653/v1/2022.findings-acl.165).
