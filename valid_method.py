import argparse
import csv
import numpy as np
import torch
import pickle as pkl
import pandas as pd
from utils import alt_tqa_evaluate, get_cluster_idxs, get_top_heads_cluster, get_separated_activations, get_interventions_dict
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset(path):
    questions = pd.read_csv(path)
    questions.dropna(axis=1, how='all', inplace=True)  # drop all-null columns
    return questions

def get_intervensions(directions, num_heads, num_layers):
    interventions = {}
    for layer in range(num_layers):
        if layer == 15:
            interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
            for head in range(num_heads):
                interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, directions[layer, head, :]))
    return interventions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf')
    parser.add_argument('--dataset_path', type=str, default=f'./AT/TruthfulQA/data/v0/TruthfulQA.csv')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--probe_base_weight', type=int, default=1)
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--num_top_heads_2_intervene', type=int, default=24)
    parser.add_argument('--dataset_name', type=str, default='mc_task.json')
    parser.add_argument('--n_clusters', type=int, default=3)
    args = parser.parse_args()

    model_path = f'./model/{args.model_name}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Running on model: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()

    # ['Type', 'Category', 'Question', 'Best Answer', 'Correct Answers', 'Incorrect Answers', 'Source']
    print('Loading dataset...')
    dataset = load_dataset(args.dataset_path)
    directions = pkl.load(open(f'./AT/directions/{args.model_name}_directions_new.pkl', 'rb'))

    positive = np.array(pkl.load(open(f'./AT/activations/Llama-2-7b-chat-hf_head_wise_positive.pkl', 'rb')))
    negative = np.array(pkl.load(open(f'./AT/activations/Llama-2-7b-chat-hf_head_wise_negative.pkl', 'rb')))
    head_wise_activations = np.concatenate([positive, negative], axis=0)
    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], head_wise_activations.shape[1], 32, 128)

    fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)
    separated_head_wise_activations, separated_labels = get_separated_activations(args)

    for i in range(args.num_fold):
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        cluster_idxs = get_cluster_idxs(args.num_layers, args.num_heads, train_set_idxs, val_set_idxs, n_clusters=args.n_clusters, directions=directions)
        top_heads, probes = get_top_heads_cluster(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, args.num_layers, args.num_heads, 
                                                  args.seed, args.num_top_heads_2_intervene, cluster_idxs, use_random_dir=False)
        
        interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads=32, threshold=0.5)

        def lt_modulated_cluster_probe_add(head_output, layer_name, start_edit_location, interventions):
            b, l, hd = head_output.shape
            h = 32
            d = hd // h
            head_output = head_output.reshape(b, l, h, d)
            for head, direction, _ in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                weight = 1 + args.probe_base_weight

                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * direction_to_add * weight
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * direction_to_add * weight
                
            head_output = head_output.reshape(b, l, h * d)
            return head_output

        curr_fold_results = alt_tqa_evaluate(
            dataset=dataset, 
            model=model, 
            tokenizer=tokenizer, 
            metric_names=['mc'], 
            verbose=False, 
            preset='qa',
            interventions=interventions,
            intervention_fn=lt_modulated_cluster_probe_add, 
        )
        
        print(curr_fold_results)
        print("="*50)


if __name__ == '__main__':
    main()
