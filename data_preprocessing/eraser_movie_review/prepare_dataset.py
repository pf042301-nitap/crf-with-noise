import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type = str, default = "movies")
    parser.add_argument("--save_path", type = str, default = "original")
    parser.add_argument("--split_tests", action = "store_true")
    parser.add_argument("--max_seq_len", type = int, default = 400)
    return parser.parse_args()


def to_lower(tokens):
    return [token.lower() for token in tokens]


def main(args):
    os.makedirs(args.save_path, exist_ok = True)
    for split, split_tgt in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        doc2rationale_range = {}
        with open(os.path.join(args.data_path, f'{split}.jsonl'), 'r') as f:
            for line in f:
                doc_info = json.loads(line)
                if len(doc_info['evidences']) > 0:
                    doc2rationale_range[doc_info['annotation_id']] = sorted([[e['start_token'], e['end_token']] for ev in doc_info['evidences'] for e in ev], key = lambda x: x[0])
                else:
                    doc2rationale_range[doc_info['annotation_id']] = []

        save_path = os.path.join(args.save_path, f"{split_tgt}.jsonl")
        print(f"Saving to ==> {save_path}")
        with open(save_path, 'w') as fw:
            for doc in os.scandir(os.path.join(args.data_path, 'docs')):
                with open(doc.path, 'r') as fr:
                    if doc.name not in doc2rationale_range:
                        continue
                    sents = fr.read().splitlines()
                    tokens = to_lower([token for sent in sents for token in sent.split(" ")])
                    rationale_ranges = doc2rationale_range[doc.name]
                    label = 1 if doc.name.startswith('pos') else 0
                    if split == "test":
                        fw.write(json.dumps([tokens, label, rationale_ranges]) + '\n')
                    else:
                        fw.write(json.dumps([tokens, label]) + '\n')

    if args.split_tests:
        with open(os.path.join(args.save_path, "test.jsonl"), "r") as f:
            test_examples = [json.loads(l) for l in f.read().splitlines()]

        split_test_examples = []
        for tokens, label, rationale_ranges in test_examples:

            while len(tokens) > args.max_seq_len:
                split_tokens = tokens[:args.max_seq_len]
                remaining_tokens = tokens[args.max_seq_len:]

                split_ranges = []
                remaining_ranges = []
                for rat_range in rationale_ranges:
                    if rat_range[1] <= args.max_seq_len:
                        split_ranges.append(rat_range)
                    elif rat_range[0] < args.max_seq_len and rat_range[1] > args.max_seq_len:
                        split_ranges.append([rat_range[0], args.max_seq_len])
                        remaining_ranges.append([0, rat_range[1] - args.max_seq_len])
                    else:
                        remaining_ranges.append([rat_range[0] - args.max_seq_len, rat_range[1] - args.max_seq_len])

                split_test_examples.append([split_tokens, label, split_ranges])

                tokens = remaining_tokens
                rationale_ranges = remaining_ranges

        save_path = os.path.join(args.save_path, "test.jsonl")
        print(f"Saving to ==> {save_path}")
        with open(save_path, "w") as f:
            for example in split_test_examples:
                f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
        main(parse_args())
