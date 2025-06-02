import argparse
import json
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import get_llama_activations_bau

def load_dataset(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

def dataset_process(dataset):
    if isinstance(dataset, list):
        questions = []
        choices = {
            'positive': [],
            'negative': [],
        }
        prompts = {
            'positive': [],
            'negative': [],
        }
        for i in range(len(dataset)):
            questions.append(dataset[i]['question'])
            for key, value in dataset[i]['mc2_targets'].items():
                prompt = f"Q: {dataset[i]['question']} A: {key}"
                if value == 1:
                    choices['positive'].append(key)
                    prompts['positive'].append(prompt)
                elif value == 0:
                    choices['negative'].append(key)
                    prompts['negative'].append(prompt)
    return questions, choices, prompts

def tokenize_prompt(prompts, tokenizer):
    tokenized_prompts = {
        'positive': [],
        'negative': [],
    }
    for kind in prompts:
        for p in prompts[kind]:
            tokenized_prompts[kind].append(tokenizer(p, return_tensors = 'pt').input_ids)
    return tokenized_prompts

def get_ac():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf')
    parser.add_argument('--dataset_name', type=str, default='mc_task.json')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = f"./model/{args.model_name}"
    print(f'Running on model: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()

    dataset_path = f'./AT/TruthfulQA/data/v0/{args.dataset_name}'
    print(f'Loading dataset: {dataset_path}')
    dataset = load_dataset(f'./AT/TruthfulQA/data/v0/{args.dataset_name}')
    _, _, prompts = dataset_process(dataset)

    print("Tokenizing")
    tokenized_prompts = tokenize_prompt(prompts, tokenizer)

    print("Getting activations")
    for kind, sub_class_prompts in tokenized_prompts.items():
        all_layer_wise_activations = []
        all_head_wise_activations = []
        for prompt in tqdm(sub_class_prompts):
            # layer_wise_activations (33, _, 4096) num_hidden_layers + last, seq_len, hidden_size
            # head_wise_activations (32, _, 4096) num_hidden_layers, seq_len, hidden_size
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
            # store layers' last token output
            all_layer_wise_activations.append(layer_wise_activations[1:, -1, :])
            all_head_wise_activations.append(head_wise_activations[:, -1, :])

        print("Saving layer wise activations")
        pickle.dump(all_layer_wise_activations, open(f'./AT/activations/{args.model_name}_layer_wise_{kind}.pkl', 'wb'))
        
        print("Saving head wise activations")
        pickle.dump(all_head_wise_activations, open(f'./AT/activations/{args.model_name}_head_wise_{kind}.pkl', 'wb'))

    print("All saved successfully")

    return 

if __name__ == '__main__':
    get_ac()