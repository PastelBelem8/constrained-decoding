This folder contains the set of experiments referring to the evaluation of Pythia models.
Our evaluation has the following structure <seed><continuation>. 
We will differentiate the analysis into 6 experiments where <seed> sampled from the data or the model, and <continuation> is either a data continuation or a model generation. 
Additionally, we will consider the examples where the <seed> either satisfies some condition or not, we will refer to these settings as conditional and unconditional.

## General experimental setup

Here we discuss experiment design decisions, including:
1. what datasets, why and how many samples we consider 
3. what models, what decoding algorithms
2. properties we're evaluating and why they are relevant to our analysis (e.g., which use cases), 


### Datasets

We currently focus on PILE, since it's a large open source dataset that has been used to train open-source versions of GPT-like models, including GPT-J, GPT-NEO, and OPT.
We also include RealToxicityPrompts dataset in our analysis, since some of the prefixes come from OpenWebText corpus (which is a subset of PILE) but provides annotations in terms of toxic behaviors. 
Particularly, we're interested in analysing the prefixes that, while not being toxic, lead to highly toxic generations.

### Models

Given the data and our interest in grounding the analysis to the pretraining set, we mainly focus our initial analysis in the subset of models pretrained on PILE.
Such models include Pythia family of models (based on GPT-NEOX), gpt-j, and GPT-NEO models.

For each model we evaluate different **decoding algorithms**, including:

- multinomial sampling (or ancestral sampling), where we sample from the model distribution.
- temperature (values in [0.1, 0.3, 0.5, 0.95, 1.15, 1.5])
- nucleus sampling or top-p (values in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
- top-k sampling (values in [2, 10, 40, 100])


### Properties

In terms of the properties, we are interested in evaluating whether different circumstances require different evaluation methodologies. 
In other words, in a world where language modeling is so good, we need more robust evaluation methodologies.
The average behavior may no longer suit our needs for differentiating language models, and in some high-risk applications it may actually be more useful to make decisions based on worst case point estimates, including 90-th percentile. 

Examples of those applications can be:
- education: chatbots for educational purposes, we'd like to minimize the chances that the model produces toxic behavior (property of interest: **toxicity**).
- healthcare: chatbots for helping diagnose patients, we'd like to minimize the chances that the model produces non factual behavior (property of interest: **factuality**).
- social media or customer service: we'd like the chatbots to keep people engaged (property of interest: **sentiment analysis**)
- story generation: when used for creative tasks, we'd like models to generate coherent and consistent outputs (property of interest: **coherence** and **consistency**)

#### Measuring different properties

- Toxicity:
- Factuality:
- Sentiment Analysis:
- Coherence:
- Bias:


### Experiment Overview

We've observed that under some circumstances the properties of the generated sequences is not unimodal. 
Moreover, the properties of these distributions may change with different decoding algorithms, thus rendering the use of mean point estimates for model selection/comparison irrealistic.


We hypothesize that under specific circumstances the models exhibit multi modal properties and that model selection should consider other measures (more aligned with the downstream task).
To show this is the case, we create 6 set of experiments

1. Randomly sample 1k words from the vocabulary and using that word sample sequences from the pretraining data (e.g., either PILE or Real Toxicity Prompt)
2. For each sequence in 1, generate 1 continuation using different decoding algorithms;
3. Using the words in 1, use constrained decoding to generate sequences from the models that satisfy those constraints.
4. Using the list of badwords (i.e., blacklist) sample 1k sequences from the pretraining data
5. Generate continuations for the different selected prefixes.
6. Use blacklist of words to generate sequences using constrained decoding.


#### Experiment 1 - Sample 1k prefixes from pretraining dataset


#### Experiment 2 - Using prefixes, sample 1 generation from each


#### Experiment 3 -

Randomly sample sequences from model.
We will sample 1k sequences from the multinomial experiments, we've conducted before.

|         Model          | 5-token prefix (% duplicates) | 10-token prefix (% duplicates)  |
| ---------------------- | ----------------------------- | ------------------------------- |
| EleutherAI/pythia-70m  |        50.82                |       7.90                   |
| EleutherAI/pythia-70m-deduped | 29.79                        |  5.48                         |
| EleutherAI/pythia-1.4b |            54.60              |             13.98               |
| EleutherAI/pythia-1.4b-deduped |      29.79                    |     5.48                       |


Selecting 10 tokens, we obtain 13.98% of duplicate prefixes, whereas with


python -m run_generate --config ../configs/generate-pythia1.4b_temp.yaml --model_name EleutherAI/pythia-1.4b
python -m run_generate --config ../configs/generate-pythia1.4b_top_p.yaml --model_name EleutherAI/pythia-1.4b --device cuda:1
python -m run_generate --config ../configs/generate-pythia1.4b_top_k.yaml --model_name EleutherAI/pythia-1.4b --device cuda:2

(changed the path in ../configs/generate-pythia1.4b_temp.yaml to reflect change in model name)
python -m run_generate --config ../configs/generate-pythia1.4b_temp.yaml --model_name EleutherAI/pythia-1.4b-deduped --device cuda:4
python -m run_generate --config ../configs/generate-pythia1.4b_top_p.yaml --model_name EleutherAI/pythia-1.4b-deduped --device cuda:5
python -m run_generate --config ../configs/generate-pythia1.4b_top_k.yaml --model_name EleutherAI/pythia-1.4b-deduped --device cuda:6

#### Experiment 4 - 

We will use two types of constraints, 
1. words from the list of [dirty, naughty and bad](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en). We expect these outputs to be very biased.
2. use target attributes, socio-demographic groups


https://huggingface.co/docs/transformers/v4.19.2/en/internal/generation_utils#transformers.NoBadWordsLogitsProcessor

#### Experiment 5 -


#### Experiment 6 - 
