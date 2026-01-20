import os
import json
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--imdb_movies", type = str, default = "aclImdb")
    parser.add_argument("--eraser_movies", type = str, default = "movies")
    parser.add_argument("--random_state", type = int, default = 2022)
    parser.add_argument("--save_path", type = str, default = "original")
    parser.add_argument("--validation_size", type = float, default = 0.20)
    return parser.parse_args()


def to_lower(tokens):
    return [token.lower() for token in tokens]


def main(args):
    os.makedirs(args.save_path, exist_ok = True)

    print("Tokenizing IMDB Movie Review...")
    examples = []
    for split in ['train', 'test']:
        for label in ['pos', 'neg']:
            for doc in tqdm(os.scandir(os.path.join(args.imdb_movies, split, label))):
                with open(doc.path, 'r') as f:
                    doc_tokens = to_lower(word_tokenize(f.read(), language = 'english', preserve_line = False))
                    doc_label = 1 if label == 'pos' else 0
                    examples.append([doc_tokens, doc_label])

    print("Splitting into train and validation...")
    train, valid = train_test_split(examples, test_size = args.validation_size, random_state = args.random_state, shuffle = True)

    save_path = os.path.join(args.save_path, "train.jsonl")
    print(f"Saving train into ==> {save_path}")
    with open(save_path, 'w') as f:
        for example in train:
            f.write(json.dumps(example) + '\n')

    save_path = os.path.join(args.save_path, "valid.jsonl")
    print(f"Saving validation into ==> {save_path}")
    with open(save_path, 'w') as f:
        for example in valid:
            f.write(json.dumps(example) + '\n')

    print("Building rationale ranges for test (Eraser Movie Review)...")
    doc2rationale_range = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(args.eraser_movies, f'{split}.jsonl'), 'r') as f:
            for line in f:
                doc_info = json.loads(line)
                if len(doc_info['evidences']) > 0:
                    doc2rationale_range[doc_info['annotation_id']] = sorted([[e['start_token'], e['end_token']] for ev in doc_info['evidences'] for e in ev], key = lambda x: x[0])
                else:
                    doc2rationale_range[doc_info['annotation_id']] = []

    save_path = os.path.join(args.save_path, "test.jsonl")
    print(f"Saving test into ==> {save_path}")
    with open(save_path, 'w') as fw:
        for doc in tqdm(os.scandir(os.path.join(args.eraser_movies, 'docs'))):
            with open(doc.path, 'r') as fr:
                if doc.name not in doc2rationale_range:
                    print(doc.name)
                    continue
                sents = fr.read().splitlines()
                tokens = to_lower([token for sent in sents for token in sent.split(" ")])
                rationale_ranges = doc2rationale_range[doc.name]
                label = 1 if doc.name.startswith('pos') else 0
                fw.write(json.dumps([tokens, label, rationale_ranges]) + '\n')


if __name__ == "__main__":
    main(parse_args())