# movies.py - Keep this file mostly unchanged, just ensure it works with the updated system
import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class MovieDataset(Dataset):
    def __init__(self, data_path, split):
        with open(os.path.join(data_path, f"{split}.jsonl"), "r") as f:
            self.data = [json.loads(line) for line in f.read().splitlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MovieDatasetWithReplacementProbs(MovieDataset):
    def __init__(self, data_path, split, noise_p):
        super().__init__(data_path = data_path, split = split)
        with open(os.path.join(data_path, "word_statistics", f"{split}_replacement_probs.pkl"), "rb") as f:
            self.replacement_probs = pickle.load(f)
        # Finish precomputing replacement probabilities
        for i, replacement_probs in enumerate(self.replacement_probs):
            replacement_probs *= noise_p
            replacement_probs[replacement_probs > 1] = 1
            self.replacement_probs[i] = replacement_probs

    def __getitem__(self, idx):
        return self.data[idx] + [self.replacement_probs[idx]]

@dataclass
class MovieDatasetFactory:
    data_path: str
    split: str
    noise_p: Optional[int] = None

    def create_dataset(self, inject_noise):
        if inject_noise:
            return MovieDatasetWithReplacementProbs(self.data_path, self.split, self.noise_p)
        return MovieDataset(self.data_path, self.split)

@dataclass
class ReviewCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[bool] = True

    def __call__(self, data):
        reviews, labels = zip(*data)
        labels_bb = torch.tensor(labels, dtype=torch.long)
        labels_rp = torch.tensor(labels, dtype=torch.long)
        reviews_tokenized = self.collate_reviews(reviews)
        return ReviewBatch(
            reviews_tokenized = reviews_tokenized,
            labels_bb = labels_bb,
            labels_rp = labels_rp
        )

    def collate_reviews(self, reviews):
        return self.tokenizer(
            text = list(reviews),
            is_split_into_words = True,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )

@dataclass
class ReviewCollatorWithReplacementProbs(ReviewCollator):
    def __call__(self, data):
        reviews, labels, replacement_probs = zip(*data)
        labels_bb = torch.tensor(labels, dtype=torch.long)
        labels_rp = torch.tensor(labels, dtype=torch.long)
        reviews_tokenized = self.collate_reviews(reviews)
        return ReviewBatch(
            reviews_tokenized = reviews_tokenized,
            labels_bb = labels_bb,
            labels_rp = labels_rp,
            reviews = reviews,
            replacement_probs = replacement_probs
        )

@dataclass
class AnnotatedReviewCollator(ReviewCollator):
    def __call__(self, data):
        reviews, labels, rationale_ranges = zip(*data)
        reviews_tokenized = self.collate_reviews(reviews)
        return AnnotatedReviewBatch(
            reviews = reviews,
            reviews_tokenized = reviews_tokenized,
            labels = labels,
            rationale_ranges = rationale_ranges
        )

@dataclass
class AnnotatedReviewCollatorWithReplacementProbs(ReviewCollator):
    def __call__(self, data):
        reviews, labels, rationale_ranges, replacement_probs = zip(*data)
        reviews_tokenized = self.collate_reviews(reviews)
        return AnnotatedReviewBatch(
            reviews = reviews,
            reviews_tokenized = reviews_tokenized,
            labels = labels,
            rationale_ranges = rationale_ranges,
            replacement_probs = replacement_probs
        )

@dataclass
class CollatorFactory:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[bool] = True

    def create_collator(self, split, inject_noise):
        if split == "test":
            if inject_noise:
                return AnnotatedReviewCollatorWithReplacementProbs(
                    tokenizer = self.tokenizer,
                    max_length = self.max_length,
                    padding = self.padding,
                    truncation = self.truncation
                )
            return AnnotatedReviewCollator(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        if inject_noise:
            return ReviewCollatorWithReplacementProbs(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        return ReviewCollator(
            tokenizer = self.tokenizer,
            max_length = self.max_length,
            padding = self.padding,
            truncation = self.truncation
        )

@dataclass
class ReviewBatch:
    reviews_tokenized: torch.Tensor
    labels_bb: torch.Tensor
    labels_rp: torch.Tensor
    reviews: Optional[tuple] = None
    replacement_probs: Optional[np.array] = None

@dataclass
class AnnotatedReviewBatch:
    reviews: tuple
    reviews_tokenized: torch.Tensor
    labels: torch.Tensor
    rationale_ranges: tuple
    replacement_probs: Optional[np.array] = None

@dataclass
class DataLoaderFactory:
    data_path: str
    noise_p: int
    batch_size: int
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    shuffle: Optional[bool] = True

    def create_dataloader(self, split, inject_noise):
        dataset = MovieDatasetFactory(self.data_path, split, self.noise_p).create_dataset(inject_noise)
        collate_fn = CollatorFactory(self.tokenizer, self.max_length).create_collator(split, inject_noise)

        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            collate_fn = collate_fn,
            shuffle = self.shuffle,
            num_workers = 1
        )