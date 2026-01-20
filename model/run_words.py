# file name: run_words.py
import json
import os
import csv
from argparse import ArgumentParser

import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from termcolor import colored
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import (BlackBoxPredictor, RationaleExtractor,
                    RationaleExtractorFactory, RationalePredictor,
                    SelectorFactory)
from movies import DataLoaderFactory

logging.set_verbosity_error()

def get_device(device_str=None):
    """Get the best available device"""
    if device_str is not None:
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_str == 'cpu':
            return torch.device('cpu')
        elif device_str.startswith('cuda:'):
            gpu_id = int(device_str.split(':')[1])
            if gpu_id < torch.cuda.device_count():
                return torch.device(f'cuda:{gpu_id}')
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")
    
    return device

def parse_args():
    parser = ArgumentParser()
    # Whether to train and/or evaluate
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--evaluate", action = "store_true")
    # Whether to inject noise
    parser.add_argument("--inject_noise", action = "store_true")
    # Magnitude of augmentation hyperparameter
    parser.add_argument('--noise_p', type = float, default = 0.1)
    # Device
    parser.add_argument('--device', type = str, default = 'cuda')
    # Optimizer BB: pytorch optim.Adam defaults
    parser.add_argument('--bb_lr', type = float, default = 2e-5)
    # Optimizer RP pytorch optim.Adam defaults
    parser.add_argument('--rp_lr', type = float, default = 2e-5)
    parser.add_argument('--lambda_sparse', type=float, default=0.01)
    parser.add_argument('--lambda_entropy', type=float, default=0.05)

    # Freeze BERT weights
    parser.add_argument('--freeze_encoder_bb', action = "store_true")
    parser.add_argument('--freeze_encoder_rp', action = "store_true")
    # Use CRF for rationale selection
    parser.add_argument('--use_crf', action = "store_true", help="Use CRF for structured rationale selection")
    # Training
    parser.add_argument('--num_epochs', type = int, default = 5)
    # Patience
    parser.add_argument('--patience', type = int, default = 2)
    # Model proximity hyperparameter
    parser.add_argument('--proximity', type = float, default = 0.1)
    # Model
    parser.add_argument('--save_path', type = str, default = os.path.join("trained", "ours"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--batch_size', type = int, default = 16)
    # Rationale Extraction hyperparameter
    parser.add_argument('--sparsity', type = float, default = 0.2)
    # Dataset
    parser.add_argument('--data_path', type = str, default = os.path.join("..", "..", "rnp_movie_review", "original"))
    # Selection method
    parser.add_argument('--selection_method', choices = ['words', 'span'], default = 'words')
    # Eval-related
    # Compare model-generated and hand-labeled rationales
    parser.add_argument('--show_detail', action = "store_true")
    parser.add_argument('--generate_csv', action="store_true")
    return parser.parse_args()

def get_param_string(noise_p, proximity, sparsity, inject_noise, freeze_encoder_bb, freeze_encoder_rp, use_crf=False):
    """Create a standardized parameter string for filenames"""
    noise_suffix = "_noise" if inject_noise else ""
    freeze_bb_suffix = "_fbb" if freeze_encoder_bb else ""
    freeze_rp_suffix = "_frp" if freeze_encoder_rp else ""
    crf_suffix = "_crf" if use_crf else ""
    return f"n_{noise_p}_prox_{proximity}_sparse_{sparsity}{noise_suffix}{freeze_bb_suffix}{freeze_rp_suffix}{crf_suffix}"

def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return
     # Get device (auto-detect if not specified)
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Save path: {args.save_path}")
    print(f"Inject noise: {args.inject_noise}")
    print(f"Noise p: {args.noise_p}")
    print(f'proximity: {args.proximity}')
    print(f'sparsity: {args.sparsity}')
    print(f'batch_size: {args.batch_size}')
    print(f'freeze_encoder_bb: {args.freeze_encoder_bb}')
    print(f'freeze_encoder_rp: {args.freeze_encoder_rp}')
    print(f'use_crf: {args.use_crf}')

    checkpoint_dir = os.path.join(args.save_path, "checkpoints")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)

    bb_model = BlackBoxPredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_bb, use_crf=args.use_crf).to(device)
    print(f"Black Box Predictor: {get_num_params(bb_model)} parameters")
    if args.freeze_encoder_bb:
        print("  BERT encoder is FROZEN in Black Box Predictor")
    if args.use_crf:
        print("  Using CRF for structured rationale selection")

    rp_model = RationalePredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_rp).to(device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")
    if args.freeze_encoder_rp:
        print("  BERT encoder is FROZEN in Rationale Predictor")

    rationale_selector = SelectorFactory(args.sparsity, args.max_length, tokenizer.pad_token_id, device).create_selector(args.selection_method, use_crf=args.use_crf)

    rationale_extractor = RationaleExtractorFactory(tokenizer, device, args.data_path).create_extractor(args.inject_noise)

    if args.train:
        os.makedirs(checkpoint_dir, exist_ok = True)
        train_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True
        ).create_dataloader("train", args.inject_noise)
        valid_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True
        ).create_dataloader("valid", args.inject_noise)

        bb_optimizer = optim.Adam(bb_model.parameters(), args.bb_lr)
        rp_optimizer = optim.Adam(rp_model.parameters(), args.rp_lr)

        validation_rationale_extractor = RationaleExtractor(tokenizer, device)

        train(
            bb_model = bb_model,
            bb_optimizer = bb_optimizer,
            rp_model = rp_model,
            rp_optimizer = rp_optimizer,
            train_loader = train_loader,
            valid_loader = valid_loader,
            eval_every = len(train_loader),
            device = device,
            proximity = args.proximity,
            num_epochs = args.num_epochs,
            patience = args.patience,
            checkpoint_dir = checkpoint_dir,
            rationale_selector = rationale_selector,
            rationale_extractor = rationale_extractor,
            validation_rationale_extractor = validation_rationale_extractor,
            noise_p=args.noise_p,
            inject_noise=args.inject_noise,
            sparsity=args.sparsity,
            freeze_encoder_bb=args.freeze_encoder_bb,
            freeze_encoder_rp=args.freeze_encoder_rp,
            use_crf=args.use_crf,
            tokenizer=tokenizer,
        )

    if args.evaluate:
        test_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = False
        ).create_dataloader("test", False)

        test_rationale_extractor = RationaleExtractor(tokenizer, device)

        evaluate(
            bb_model = bb_model,
            rp_model = rp_model,
            tokenizer = tokenizer,
            test_loader = test_loader,
            device = device,
            show_detail = args.show_detail,
            rationale_selector = rationale_selector,
            rationale_extractor = test_rationale_extractor,
            checkpoint_dir = checkpoint_dir,
            result_path = args.save_path,
            proximity=args.proximity,
            noise_p=args.noise_p,
            inject_noise=args.inject_noise,
            sparsity=args.sparsity,
            freeze_encoder_bb=args.freeze_encoder_bb,
            freeze_encoder_rp=args.freeze_encoder_rp,
            use_crf=args.use_crf,
            generate_csv=args.generate_csv,
        )

def train(
    bb_model,
    bb_optimizer,
    rp_model,
    rp_optimizer,
    train_loader,
    valid_loader,
    eval_every,
    device,
    proximity,
    num_epochs,
    patience,
    checkpoint_dir,
    rationale_selector,
    rationale_extractor,
    validation_rationale_extractor,
    noise_p,
    inject_noise,
    sparsity,
    freeze_encoder_bb,
    freeze_encoder_rp,
    use_crf=False,
    tokenizer=None,
    ):

    print(f"Training with {len(train_loader)} batches, eval_every={eval_every}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    print(f"Freeze encoder BB: {freeze_encoder_bb}")
    print(f"Freeze encoder RP: {freeze_encoder_rp}")
    print(f"Use CRF: {use_crf}")
    
    # Create param_string early
    param_string = get_param_string(noise_p, proximity, sparsity, inject_noise, freeze_encoder_bb, freeze_encoder_rp, use_crf)
    
    with tqdm(total=num_epochs * len(train_loader)) as pb:

        # Initialize statistics
        bb_running_train_loss = 0.0
        bb_best_train_loss = float("Inf")
        rp_running_train_loss = 0.0
        rp_best_train_loss = float("Inf")
        bb_running_valid_loss = 0.0
        bb_best_valid_loss = float("Inf")
        rp_running_valid_loss = 0.0
        rp_best_valid_loss = float("Inf")
        running_train_replace_ratio = 0.0
        running_valid_replace_ratio = 0.0
        global_step = 0
        metrics = []
        patience_left = patience

        # training loop
        bb_model.train()
        rp_model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:

                if patience_left == 0:
                    pb.write("Patience is 0, early stopping")
                    break

                # generate prediction and token probs of being in a rationale
                batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                
                if use_crf:
                    att_pred, token_att, emission_scores, crf_tags = bb_model(
                        **batch.reviews_tokenized,
                        return_crf_scores=True
                    )
                    
                    # Create mask for CRF loss
                    mask = (batch.reviews_tokenized.input_ids[:, 1:] != 0) & \
                           (batch.reviews_tokenized.attention_mask[:, 1:] == 1)
                    
                    hard_mask = rationale_selector(
                        token_att = token_att,
                        input_ids = batch.reviews_tokenized.input_ids,
                        crf_tags = crf_tags
                    )
                else:
                    att_pred, token_att, _ = bb_model(**batch.reviews_tokenized)
                    
                    hard_mask = rationale_selector(
                        token_att = token_att,
                        input_ids = batch.reviews_tokenized.input_ids
                    )
                    emission_scores = None
                    crf_tags = None
                    mask = None
                
                rationale, _, replace_ratio = rationale_extractor(
                    batch = batch,
                    hard_mask = hard_mask
                )

                # predict from rationale
                hard_pred = rp_model(**rationale)
            
                if use_crf:
                    bb_loss = bb_model.get_loss(
                        att_pred=att_pred,
                        hard_pred=hard_pred.detach(),
                        labels=batch.labels_bb.to(device),
                        proximity=proximity,
                        emission_scores=emission_scores,
                        token_att=token_att,
                        mask=mask
                    )

                else:
                    bb_loss = bb_model.get_loss(
                        att_pred = att_pred,
                        hard_pred = hard_pred.detach(),
                        labels = batch.labels_bb.to(device),
                        proximity = proximity
                    )

                rp_loss = rp_model.get_loss(
                    att_pred = att_pred.detach(),
                    hard_pred = hard_pred,
                    labels = batch.labels_rp.to(device),
                    proximity = proximity
                )

                # Check for NaN losses
                if torch.isnan(bb_loss) or torch.isnan(rp_loss):
                    print(f"Warning: NaN loss detected. BB Loss: {bb_loss}, RP Loss: {rp_loss}")
                    # Skip this batch if loss is NaN
                    continue

                bb_optimizer.zero_grad()
                rp_optimizer.zero_grad()

                bb_loss.backward()
                rp_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(bb_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(rp_model.parameters(), max_norm=1.0)

                bb_optimizer.step()
                rp_optimizer.step()
            
                pb.update(1)

                # update running values
                bb_running_train_loss += bb_loss.item()
                rp_running_train_loss += rp_loss.item()
                running_train_replace_ratio += replace_ratio
                global_step += 1

                # validation step
                if global_step % eval_every == 0:
                    bb_model.eval()
                    rp_model.eval()
                    with torch.no_grad():                    
                        for val_batch in valid_loader:
                            # generate prediction and token probs of being in a rationale
                            val_batch.reviews_tokenized = val_batch.reviews_tokenized.to(device)
                            
                            if use_crf:
                                val_att_pred, val_token_att, val_emission_scores, val_crf_tags = bb_model(
                                    **val_batch.reviews_tokenized,
                                    return_crf_scores=True
                                )
                                
                                val_hard_mask = rationale_selector(
                                    token_att = val_token_att,
                                    input_ids = val_batch.reviews_tokenized.input_ids,
                                    crf_tags = val_crf_tags
                                )
                            else:
                                val_att_pred, val_token_att, _ = bb_model(**val_batch.reviews_tokenized)
                                
                                val_hard_mask = rationale_selector(
                                    token_att = val_token_att,
                                    input_ids = val_batch.reviews_tokenized.input_ids
                                )

                            val_rationale, _, val_replace_ratio = validation_rationale_extractor(
                                batch = val_batch,
                                hard_mask = val_hard_mask
                            )

                            # predict from rationale
                            val_hard_pred = rp_model(**val_rationale)
            
                            if use_crf:
                                bb_valid = bb_model.get_loss(
                                    att_pred = val_att_pred,
                                    hard_pred = val_hard_pred,
                                    labels = val_batch.labels_bb.to(device),
                                    proximity = proximity
                                )
                            else:
                                bb_valid = bb_model.get_loss(
                                    att_pred = val_att_pred,
                                    hard_pred = val_hard_pred,
                                    labels = val_batch.labels_bb.to(device),
                                    proximity = proximity
                                )

                            rp_valid = rp_model.get_loss(
                                att_pred = val_att_pred,
                                hard_pred = val_hard_pred,
                                labels = val_batch.labels_rp.to(device),
                                proximity = proximity
                            )

                            # Check for NaN validation losses
                            if torch.isnan(bb_valid) or torch.isnan(rp_valid):
                                print(f"Warning: NaN validation loss detected")
                                continue

                            bb_running_valid_loss += bb_valid.item()
                            rp_running_valid_loss += rp_valid.item()
                            running_valid_replace_ratio += val_replace_ratio

                    # evaluation
                    bb_average_train_loss = bb_running_train_loss / eval_every
                    rp_average_train_loss = rp_running_train_loss / eval_every
                    average_train_replace_ratio = running_train_replace_ratio / eval_every

                    bb_average_valid_loss = bb_running_valid_loss / len(valid_loader)
                    rp_average_valid_loss = rp_running_valid_loss / len(valid_loader)
                    average_valid_replace_ratio = running_valid_replace_ratio / len(valid_loader)

                    bb_improved = bb_best_valid_loss > bb_average_valid_loss
                    rp_improved = rp_best_valid_loss > rp_average_valid_loss

                    patience_left = patience if bb_improved and rp_improved else patience_left - 1

                    metrics.append({
                        "bb": {
                            "train_loss": bb_average_train_loss,
                            "valid_loss": bb_average_valid_loss
                        },
                        "rp": {
                            "train_loss": rp_average_train_loss,
                            "valid_loss": rp_average_valid_loss
                        },
                        "replace_ratio": {
                            "replace_train_ratio": average_train_replace_ratio,
                            "replace_valid_ratio": average_valid_replace_ratio
                        },
                        "patience_left": patience_left,
                        "step": global_step,
                        "epoch": epoch + 1,
                        "freeze_encoder_bb": freeze_encoder_bb,
                        "freeze_encoder_rp": freeze_encoder_rp,
                        "use_crf": use_crf,
                    })

                    # update running values
                    if not torch.isnan(torch.tensor(bb_average_train_loss)):
                        bb_best_train_loss = min(bb_best_train_loss, bb_average_train_loss)
                    if not torch.isnan(torch.tensor(bb_average_valid_loss)):
                        bb_best_valid_loss = min(bb_best_valid_loss, bb_average_valid_loss)
                    if not torch.isnan(torch.tensor(rp_average_train_loss)):
                        rp_best_train_loss = min(rp_best_train_loss, rp_average_train_loss)
                    if not torch.isnan(torch.tensor(rp_average_valid_loss)):
                        rp_best_valid_loss = min(rp_best_valid_loss, rp_average_valid_loss)

                    # resetting running values
                    bb_running_train_loss = 0.0
                    rp_running_train_loss = 0.0
                    running_train_replace_ratio = 0.0
                    bb_running_valid_loss = 0.0
                    rp_running_valid_loss = 0.0
                    running_valid_replace_ratio = 0.0

                    # print progress
                    pb.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}]')
                    pb.write(f'Train Probability of Replacement: {average_train_replace_ratio * 100:.4f}')
                    pb.write(f'Valid Probability of Replacement: {average_valid_replace_ratio * 100:.4f}')
                    pb.write(f'BB Train Loss: {bb_average_train_loss:.4f}, BB Valid Loss: {bb_average_valid_loss:.4f}')
                    pb.write(f'RP Train Loss: {rp_average_train_loss:.4f}, RP Valid Loss: {rp_average_valid_loss:.4f}')
                    pb.write(f"Patience: {patience_left}")
                    pb.write(f"Freeze BB Encoder: {freeze_encoder_bb}, Freeze RP Encoder: {freeze_encoder_rp}")
                    pb.write(f"Use CRF: {use_crf}")

                    # checkpoint 
                    if bb_improved and rp_improved:
                        pb.write(f'Model saved to ==> {bb_model_save(bb_model, checkpoint_dir, param_string)}')
                        pb.write(f'Model saved to ==> {rp_model_save(rp_model, checkpoint_dir, param_string)}')
                    pb.write(f'Metrics saved to ==> {metrics_save(metrics, checkpoint_dir, param_string)}')

                    bb_model.train()
                    rp_model.train()

            if patience_left == 0:
                break
        print(f"\nTraining completed!")
        print(f"Best BB Validation Loss: {bb_best_valid_loss:.4f}")
        print(f"Best RP Validation Loss: {rp_best_valid_loss:.4f}")
        print(f"Freeze BB Encoder: {freeze_encoder_bb}, Freeze RP Encoder: {freeze_encoder_rp}")
        print(f"Use CRF: {use_crf}")
        
        # Final save
        bb_model_save(bb_model, checkpoint_dir, param_string)
        rp_model_save(rp_model, checkpoint_dir, param_string)
        metrics_save(metrics, checkpoint_dir, param_string)
        print(f"Final models and metrics saved")

def evaluate(
    bb_model,
    rp_model,
    tokenizer,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    checkpoint_dir,
    result_path, 
    proximity, 
    noise_p,
    inject_noise,
    sparsity,
    freeze_encoder_bb,
    freeze_encoder_rp,
    use_crf=False,
    generate_csv=False,
    ):

    if checkpoint_dir is not None:
        param_string = get_param_string(noise_p, proximity, sparsity, inject_noise, freeze_encoder_bb, freeze_encoder_rp, use_crf)
        print(f'Loading models with noise_p={noise_p}, proximity={proximity}, sparsity={sparsity}, inject_noise={inject_noise}, freeze_bb={freeze_encoder_bb}, freeze_rp={freeze_encoder_rp}, use_crf={use_crf}')
        
        try:
            bb_model_load(bb_model, checkpoint_dir, param_string, device)
        except FileNotFoundError as e:
            print(f"Could not load BB model: {e}")
            return
            
        try:
            rp_model_load(rp_model, checkpoint_dir, param_string, device)
        except FileNotFoundError as e:
            print(f"Could not load RP model: {e}")
            return
            
        print("Models loaded successfully")

    if show_detail:
        detail_path = os.path.join(os.path.dirname(checkpoint_dir), "details")
        os.makedirs(detail_path, exist_ok=True)

    # Initialize CSV data collection
    csv_data = []
    
    gen_spans = 0
    rat_spans = 0
    gen_rat_span_ratio = 0.0
    gen_rat_span_rtotal = 0
    max_gen_span = torch.zeros(1)
    max_rat_span = torch.zeros(1)

    tp = 0
    fp = 0
    fn = 0

    rratio = 0.0

    rprec = 0
    rrec = 0
    rf1 = 0
    rtotal = 0

    y_pred = []
    y_true = []

    comp = []
    suff = []

    ious = []
    num_gen_tokens = []
    num_rat_tokens = []

    review_count = 0

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Convert labels to tensor if needed
            if hasattr(batch, 'labels') and isinstance(batch.labels, tuple):
                batch.labels = torch.tensor(batch.labels)
            
            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
            
            if use_crf:
                att_pred, token_att, emission_scores, crf_tags = bb_model(
                    **batch.reviews_tokenized,
                    return_crf_scores=True
                )
                
                hard_mask = rationale_selector(
                    token_att = token_att,
                    input_ids = batch.reviews_tokenized.input_ids,
                    crf_tags = crf_tags
                )
            else:
                att_pred, token_att, _ = bb_model(**batch.reviews_tokenized)
                
                hard_mask = rationale_selector(
                    token_att = token_att,
                    input_ids = batch.reviews_tokenized.input_ids
                )
            
            # apply mask and recover rationale
            rationale, remainder, replace_ratio = rationale_extractor(batch, hard_mask)
            rratio += replace_ratio
            
            # predict from rationale
            hard_pred_logits = rp_model(**rationale)
            hard_pred_probs = torch.sigmoid(hard_pred_logits)

            predictions = torch.argmax(hard_pred_logits, 1).tolist()
            y_pred.extend(predictions)
            
            # Handle labels
            if hasattr(batch, 'labels'):
                if isinstance(batch.labels, torch.Tensor):
                    y_true.extend(batch.labels.tolist())
                else:
                    y_true.extend(list(batch.labels))

            # Only compute these metrics if we have labels
            if hasattr(batch, 'labels'):
                label_pred_probs = get_label_pred_probs(hard_pred_probs, batch.labels)

                remainder_hard_pred_probs = torch.sigmoid(rp_model(**remainder))
                remainder_label_pred_probs = get_label_pred_probs(remainder_hard_pred_probs, batch.labels)

                all_hard_pred_probs = torch.sigmoid(rp_model(**batch.reviews_tokenized))
                all_label_pred_probs = get_label_pred_probs(all_hard_pred_probs, batch.labels)

                comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
                suff.extend((all_label_pred_probs - label_pred_probs).tolist())

            for i in range(hard_mask.shape[0]):
                # Get the generated mask
                gen_mask = torch.tensor([False] + hard_mask[i, :, :].squeeze().tolist())
                
                # Get rationale mask (hand-labeled) if available
                if hasattr(batch, 'rationale_ranges'):
                    rat_mask = torch.tensor([any(id in range(low, high) for (low, high) in batch.rationale_ranges[i]) 
                                             for id in batch.reviews_tokenized.word_ids(i)])
                else:
                    # If no rationale ranges, create empty mask
                    rat_mask = torch.zeros_like(gen_mask, dtype=torch.bool)
                
                # Calculate original review text
                review_text = " ".join(batch.reviews[i])
                
                # Extract rationale selected by model
                # Convert token mask to word-level selection
                word_ids = batch.reviews_tokenized.word_ids(i)
                gen_word_ids = set()
                for token_idx, word_id in enumerate(word_ids):
                    if token_idx < len(gen_mask) and gen_mask[token_idx] and word_id is not None:
                        gen_word_ids.add(word_id)
                
                # Get selected words
                selected_words = []
                for word_id in sorted(gen_word_ids):
                    if word_id < len(batch.reviews[i]):
                        selected_words.append(batch.reviews[i][word_id])
                rationale_selected = " ".join(selected_words)
                
                # Get original rationale if available
                original_rationale_words = []
                if hasattr(batch, 'rationale_ranges'):
                    for low, high in batch.rationale_ranges[i]:
                        if low < len(batch.reviews[i]):
                            end_idx = min(high, len(batch.reviews[i]))
                            original_rationale_words.extend(batch.reviews[i][low:end_idx])
                original_rationale = " ".join(original_rationale_words)
                
                # Get prediction and ground truth if available
                prediction = "POSITIVE" if predictions[i] == 1 else "NEGATIVE"
                ground_truth = ""
                if hasattr(batch, 'labels'):
                    if isinstance(batch.labels, torch.Tensor):
                        current_label = batch.labels[i].item()
                    else:
                        current_label = batch.labels[i]
                    ground_truth = "POSITIVE" if current_label == 1 else "NEGATIVE"
                
                # Add to CSV data
                if generate_csv:
                    csv_data.append({
                        "review_id": review_count,
                        "review": review_text,
                        "rationale_selected": rationale_selected,
                        "original_rationale": original_rationale,
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "prediction_correct": prediction == ground_truth if ground_truth else None,
                        "selected_word_count": len(selected_words),
                        "original_rationale_word_count": len(original_rationale_words)
                    })
                
                # Calculate metrics if we have rationale ranges
                if hasattr(batch, 'rationale_ranges'):
                    gen_span = torch.logical_and(gen_mask[:-1] == False, gen_mask[1:] == True).sum()
                    gen_spans += gen_span
                    max_gen_span = torch.max(torch.stack([max_gen_span.squeeze(), gen_span.squeeze()]))
                    rat_span = torch.logical_and(rat_mask[:-1] == False, rat_mask[1:] == True).sum()
                    max_rat_span = torch.max(torch.stack([max_rat_span.squeeze(), rat_span.squeeze()]))
                    rat_spans += rat_span
                    if not rat_span == torch.zeros(1):
                        gen_rat_span_ratio += gen_span/(rat_span)
                        gen_rat_span_rtotal += 1

                    rtp = torch.sum(gen_mask & rat_mask).item()
                    tp += rtp
                    rfn = torch.sum(~gen_mask & rat_mask).item()
                    fn += rfn
                    rfp = torch.sum(gen_mask & ~rat_mask).item()
                    fp += rfp

                    rtotal += 1
                    rprec += rtp/(rtp + rfp + 1e-6)
                    rrec += rtp/(rtp + rfn + 1e-6)
                    rf1 += rtp/(rtp + ((rfp + rfn)/2) + 1e-6)

                if show_detail and hasattr(batch, 'rationale_ranges'):
                    with open(os.path.join(detail_path, f"review_{review_count}.txt"), "w") as f:
                        review_tokens = tokenizer.convert_ids_to_tokens(batch.reviews_tokenized.input_ids[i])
                        review_tokens_colored = [color_token(token, g, h) for token, g, h in zip(review_tokens, gen_mask, rat_mask)]
                        print(" ".join(review_tokens_colored), file = f)
                        if hasattr(batch, 'labels'):
                            if isinstance(batch.labels, torch.Tensor):
                                current_label = batch.labels[i].item()
                            else:
                                current_label = batch.labels[i]
                            print(f"Class: {'POSITIVE' if current_label else 'NEGATIVE'}", file = f)
                        print(f"P: {100*rtp/(rtp + rfp + 1e-6):.2f} R: {100*rtp/(rtp + rfn + 1e-6):.2f} F1: {100*rtp/(rtp + ((rfp + rfn)/2) + 1e-6):.2f}", file = f)
                    
                review_count += 1

                if hasattr(batch, 'rationale_ranges'):
                    gen_sets = to_sets(to_ranges(gen_mask))
                    num_gen_tokens.append(sum([len(s) for s in gen_sets]))
                    rat_sets = to_sets(to_ranges(rat_mask))
                    num_rat_tokens.append(sum([len(s) for s in rat_sets]))

                    rious = [(max([len(gen_set & rat_set)/(len(gen_set | rat_set) + 1e-6) for rat_set in rat_sets] + [0.0]), len(gen_set)) for gen_set in gen_sets]
                    ious.append(rious)

    # Save CSV file if requested
    if generate_csv and csv_data:
        # Create result path if it doesn't exist
        os.makedirs(result_path, exist_ok=True)
        
        # Create parameterized filename
        param_string = get_param_string(noise_p, proximity, sparsity, inject_noise, freeze_encoder_bb, freeze_encoder_rp, use_crf)
        csv_path = os.path.join(result_path, f"evaluation_results_{param_string}.csv")
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['review_id', 'review', 'rationale_selected', 'original_rationale', 
                         'prediction', 'ground_truth', 'prediction_correct',
                         'selected_word_count', 'original_rationale_word_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"CSV file saved to: {csv_path}")
        print(f"Total reviews in CSV: {len(csv_data)}")
        
        # Also save a summary CSV
        summary_path = os.path.join(result_path, f"summary_{param_string}.csv")
        with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Reviews', len(csv_data)])
            if any(row['prediction_correct'] is not None for row in csv_data):
                correct_predictions = sum(1 for row in csv_data if row['prediction_correct'])
                writer.writerow(['Correct Predictions', correct_predictions])
                writer.writerow(['Accuracy', correct_predictions / len(csv_data) if len(csv_data) > 0 else 0])
            writer.writerow(['Average Selected Words', sum(row['selected_word_count'] for row in csv_data) / len(csv_data) if len(csv_data) > 0 else 0])
            writer.writerow(['Average Original Rationale Words', sum(row['original_rationale_word_count'] for row in csv_data) / len(csv_data) if len(csv_data) > 0 else 0])
        
        print(f"Summary CSV saved to: {summary_path}")

    # Calculate rationale metrics only if we have rationale ranges
    if rtotal > 0:
        micro_prec = tp/(tp + fp) if (tp + fp) > 0 else 0
        micro_rec = tp/(tp + fn) if (tp + fn) > 0 else 0
        micro_f1 = tp/(tp + ((fp + fn)/2)) if (tp + ((fp + fn)/2)) > 0 else 0
        macro_prec = rprec/rtotal if rtotal > 0 else 0
        macro_rec = rrec/rtotal if rtotal > 0 else 0
        macro_f1 = rf1/rtotal if rtotal > 0 else 0

        micro_iou = dict()
        macro_iou = dict()
        
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        for threshold in iou_thresholds:
            thresholded_ious = [sum([int(riou >= threshold) * riou_tokens for riou, riou_tokens in rious]) for rious in ious]

            micro_iou[threshold] = dict()
            micro_iou[threshold]["prec"] = sum(thresholded_ious) / sum(num_gen_tokens) if sum(num_gen_tokens) > 0 else 0
            micro_iou[threshold]["rec"] = sum(thresholded_ious) / sum(num_rat_tokens) if sum(num_rat_tokens) > 0 else 0
            if micro_iou[threshold]["prec"] + micro_iou[threshold]["rec"] > 0:
                micro_iou[threshold]["f1"] = (2 * micro_iou[threshold]["prec"] * micro_iou[threshold]["rec"])/(micro_iou[threshold]["prec"] + micro_iou[threshold]["rec"])
            else:
                micro_iou[threshold]["f1"] = 0

            iou_rprec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_gen_tokens)]
            iou_rrec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_rat_tokens)]
            macro_iou[threshold] = dict()
            macro_iou[threshold]["prec"] = sum(iou_rprec) / len(iou_rprec) if len(iou_rprec) > 0 else 0
            macro_iou[threshold]["rec"] = sum(iou_rrec) / len(iou_rrec) if len(iou_rrec) > 0 else 0
            if macro_iou[threshold]["prec"] + macro_iou[threshold]["rec"] > 0:
                macro_iou[threshold]["f1"] = (2 * macro_iou[threshold]["prec"] * macro_iou[threshold]["rec"])/(macro_iou[threshold]["prec"] + macro_iou[threshold]["rec"])
            else:
                macro_iou[threshold]["f1"] = 0

        results = {
            "rationales": {
                "micro": {"prec": micro_prec, "rec": micro_rec, "F1": micro_f1},
                "macro": {"prec": macro_prec, "rec": macro_rec, "F1": macro_f1},
            },
            "token_selector_sparsity": rationale_selector.sparsity if hasattr(rationale_selector, 'sparsity') else sparsity,
            "replace_ratio": rratio/len(test_loader) if len(test_loader) > 0 else 0,
            "comp_suff": {
                "comprehensiveness": sum(comp)/rtotal if rtotal > 0 else 0, 
                "sufficiency": sum(suff)/rtotal if rtotal > 0 else 0
            },
            "macro_iou": macro_iou,
            "micro_iou": micro_iou,
            "freeze_encoder_bb": freeze_encoder_bb,
            "freeze_encoder_rp": freeze_encoder_rp,
            "use_crf": use_crf,
            "noise_p": noise_p,
            "proximity": proximity,
            "sparsity": sparsity,
            "inject_noise": inject_noise,
        }
    else:
        results = {
            "rationales": {
                "micro": {"prec": 0, "rec": 0, "F1": 0},
                "macro": {"prec": 0, "rec": 0, "F1": 0},
            },
            "token_selector_sparsity": rationale_selector.sparsity if hasattr(rationale_selector, 'sparsity') else sparsity,
            "replace_ratio": rratio/len(test_loader) if len(test_loader) > 0 else 0,
            "comp_suff": {
                "comprehensiveness": 0, 
                "sufficiency": 0
            },
            "macro_iou": {},
            "micro_iou": {},
            "freeze_encoder_bb": freeze_encoder_bb,
            "freeze_encoder_rp": freeze_encoder_rp,
            "use_crf": use_crf,
            "noise_p": noise_p,
            "proximity": proximity,
            "sparsity": sparsity,
            "inject_noise": inject_noise,
        }

    # Add classification metrics if we have predictions
    if y_true and y_pred:
        results["accuracy"] = classification_report(y_true, y_pred, labels=[1,0], digits=4, output_dict=True).get("accuracy", 0)
        print('\nClassification Report:')
        print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    else:
        results["accuracy"] = 0

    if generate_csv:
        results["csv_generated"] = True
        results["total_samples_in_csv"] = len(csv_data)
        results["csv_file_path"] = csv_path
        results["summary_file_path"] = summary_path

    if not show_detail:
        param_string = get_param_string(noise_p, proximity, sparsity, inject_noise, freeze_encoder_bb, freeze_encoder_rp, use_crf)
        save_results(results, result_path, param_string)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Freeze encoder BB: {freeze_encoder_bb}")
    print(f"Freeze encoder RP: {freeze_encoder_rp}")
    print(f"Use CRF: {use_crf}")
    print(f"Inject Noise: {inject_noise}")
    print(f"Noise p: {noise_p}")
    print(f"Proximity: {proximity}")
    print(f"Sparsity: {sparsity}")
    
    if rtotal > 0:
        print("\nRationale Selection Performance:")
        print(f"Token-level Micro-Averaged Precision: {micro_prec:.4f} Recall: {micro_rec:.4f} F1: {micro_f1:.4f}")
        print(f"Token-level Macro-Averaged Precision: {macro_prec:.4f} Recall: {macro_rec:.4f} F1: {macro_f1:.4f}")
        
        print("\nIoU Metrics:")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if t in micro_iou:
                print(f"Threshold={t}: Micro-P: {micro_iou[t]['prec']:.4f} R: {micro_iou[t]['rec']:.4f} F1: {micro_iou[t]['f1']:.4f}")
                print(f"Threshold={t}: Macro-P: {macro_iou[t]['prec']:.4f} R: {macro_iou[t]['rec']:.4f} F1: {macro_iou[t]['f1']:.4f}")
        
        print(f"\nReplacement Ratio: {rratio/len(test_loader):.4f}")
        print(f"\nSpan Statistics:")
        print(f"Average generated spans: {gen_spans/rtotal:.4f}")
        print(f"Average labeled rationale spans: {rat_spans/rtotal:.4f}")
        print(f"Maximum generated spans: {max_gen_span:.0f}")
        print(f"Maximum labeled rationale spans: {max_rat_span:.0f}")
        print(f"Average ratio of generated to labeled spans: {gen_rat_span_ratio/gen_rat_span_rtotal if gen_rat_span_rtotal > 0 else 0:.4f}")
        
        print(f"\nComprehensiveness: {sum(comp)/rtotal if rtotal > 0 else 0:.4f}")
        print(f"Sufficiency: {sum(suff)/rtotal if rtotal > 0 else 0:.4f}")
    
    if generate_csv:
        print(f"\nCSV Results:")
        print(f"  Total reviews processed: {len(csv_data)}")
        print(f"  CSV file saved to: {csv_path}")
        print(f"  Summary file saved to: {summary_path}")


def save_results(results, result_path, param_string):
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, f"results_{param_string}.json"), "w") as f:
        json.dump(results, f, indent=2)


def to_ranges(mask):
    t1 = torch.tensor(mask.tolist() + [0], dtype=torch.bool)
    t2 = torch.tensor([0] + mask.tolist(), dtype=torch.bool)
    start = torch.logical_and(t1, ~t2)
    end = torch.logical_and(~t1, t2)
    indices = torch.arange(len(t1))
    return list(zip(indices[start].tolist(), indices[end].tolist()))


def to_sets(ranges):
    return [set(range(low, high)) for low, high in ranges]


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_label_pred_probs(pred_probs, labels):
    # Handle both tensor and tuple/list labels
    if isinstance(labels, torch.Tensor):
        return torch.tensor([pred_prob[label.item()] for pred_prob, label in zip(pred_probs, labels)])
    else:
        # Assume labels is iterable (tuple, list, etc.)
        return torch.tensor([pred_prob[label] for pred_prob, label in zip(pred_probs, labels)])


def color_token(token, generated, handlabeled):
    if generated and handlabeled:
        return colored(token, "green")
    if not generated and handlabeled:
        return colored(token, "blue")
    if generated and not handlabeled:
        return colored(token, "red")
    return token


def model_save(model, path):
    torch.save(model.state_dict(), path)


def bb_model_save(model, path, param_string):
    model_name = f"bb_model_{param_string}.pt"
    save_path = os.path.join(path, model_name)
    model_save(model, save_path)
    return save_path


def rp_model_save(model, path, param_string):
    model_name = f"rp_model_{param_string}.pt"
    save_path = os.path.join(path, model_name)
    model_save(model, save_path)
    return save_path


def model_load(model, path, device=None):
    """Load model and move to appropriate device"""
    if device is None:
        device = torch.device('cpu')
    
    # Load with map_location to handle device mismatch
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def bb_model_load(model, path, param_string, device=None):
    model_path = os.path.join(path, f"bb_model_{param_string}.pt")
    
    if os.path.exists(model_path):
        model_load(model, model_path, device)
        print(f"Loaded BB model from {model_path}")
    else:
        # Try to find any model with matching pattern
        import glob
        pattern = os.path.join(path, f"bb_model_*.pt")
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"Warning: Model with exact parameters not found. Loading first match: {matching_files[0]}")
            model_load(model, matching_files[0], device)
            print(f"Loaded BB model from {matching_files[0]}")
        else:
            raise FileNotFoundError(f"BB model file not found: {model_path}")


def rp_model_load(model, path, param_string, device=None):
    model_path = os.path.join(path, f"rp_model_{param_string}.pt")
    
    if os.path.exists(model_path):
        model_load(model, model_path, device)
        print(f"Loaded RP model from {model_path}")
    else:
        # Try to find any model with matching pattern
        import glob
        pattern = os.path.join(path, f"rp_model_*.pt")
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"Warning: Model with exact parameters not found. Loading first match: {matching_files[0]}")
            model_load(model, matching_files[0], device)
            print(f"Loaded RP model from {matching_files[0]}")
        else:
            raise FileNotFoundError(f"RP model file not found: {model_path}")


def metrics_save(metrics, path, param_string):
    save_path = os.path.join(path, f"metrics_{param_string}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return save_path


if __name__ == "__main__":
    main(parse_args())