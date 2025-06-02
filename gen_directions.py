import argparse
import json
import numpy as np
import pickle as pkl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf')
    args = parser.parse_args()

    positive_head_wise_activations = pkl.load(open(f'./AT/activations/{args.model_name}_head_wise_positive.pkl', 'rb'))
    negative_head_wise_activations = pkl.load(open(f'./AT/activations/{args.model_name}_head_wise_negative.pkl', 'rb'))

    positive_head_wise_activations = np.array(positive_head_wise_activations)
    negative_head_wise_activations = np.array(negative_head_wise_activations)

    def load_dataset(path):
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data

    dataset = load_dataset(f'./AT/TruthfulQA/data/v0/mc_task.json')

    separated_directions = []
    for i in range(len(dataset)):
        positive_activations = []
        negative_activations = []
        for v in dataset[i]['mc2_targets'].values():
            if v == 1:
                positive_activations.append(positive_head_wise_activations[i])
            elif v == 0:
                negative_activations.append(negative_head_wise_activations[i])
        positive_activations = np.array(positive_activations)
        negative_activations = np.array(negative_activations)

        positive_activations = reshape_head_wise_activations(positive_activations)
        negative_activations = reshape_head_wise_activations(negative_activations)

        separated_directions.append(positive_activations.mean(axis=0) - negative_activations.mean(axis=0))
    separated_directions = np.array(separated_directions)
    print(separated_directions.shape)
    pkl.dump(separated_directions, open(f'./AT/directions/{args.model_name}_directions_new.pkl', 'wb'))

    positive_head_wise_activations = reshape_head_wise_activations(positive_head_wise_activations)
    negative_head_wise_activations = reshape_head_wise_activations(negative_head_wise_activations)

    # generate directions for each question [32, 32, 128]
    head_wise_activation_directions = np.array(negative_head_wise_activations.mean(axis=0) - positive_head_wise_activations.mean(axis=0))
    print(head_wise_activation_directions.shape)
    pkl.dump(head_wise_activation_directions, open(f'./AT/directions/{args.model_name}_directions.pkl', 'wb'))

def reshape_head_wise_activations(head_wise_activations):
    b, l, hd = head_wise_activations.shape
    h = 32
    d = hd // h
    head_wise_activations = head_wise_activations.reshape(b, l, h, d)
    return head_wise_activations

if __name__ == '__main__':
    main()
