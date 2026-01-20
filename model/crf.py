# file name: crf.py
import torch
import torch.nn as nn

class CRFLayer(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    Uses Viterbi algorithm for decoding during inference.
    """
    def __init__(self, num_tags, batch_first=True):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrix: transition[i, j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        
        # Initialize transitions
        self.init_weights()
    
    def init_weights(self):
        # Initialize with small values
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        
        # Small bias for self-transitions
        with torch.no_grad():
            self.transitions[0, 0] += 0.5  # tag0->tag0
            self.transitions[1, 1] += 0.5  # tag1->tag1
    
    def forward(self, emissions, tags, mask=None):
        """
        Compute the negative log likelihood for given emissions and tags.
        
        Args:
            emissions: (batch_size, seq_len, num_tags) if batch_first
            tags: (batch_size, seq_len)
            mask: (batch_size, seq_len)
            
        Returns:
            log_likelihood: scalar
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        
        # Negative log likelihood
        log_likelihood = denominator - numerator
        
        # Return mean loss (scalar)
        return log_likelihood.mean()
    
    def decode(self, emissions, mask=None):
        """
        Find the most likely tag sequence using Viterbi algorithm.
        
        Args:
            emissions: (batch_size, seq_len, num_tags)
            mask: (batch_size, seq_len)
            
        Returns:
            best_tags: (batch_size, seq_len)
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        return self._viterbi_decode(emissions, mask)
    
    def _compute_score(self, emissions, tags, mask):
        """Compute the score of a given tag sequence."""
        batch_size, seq_length = tags.shape
        
        # Start with transition from start tag
        score = self.start_transitions[tags[:, 0]]
        
        # Add emission scores for first tag
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
        for i in range(1, seq_length):
            # If mask[i] is False, we should stop
            if mask is not None:
                mask_i = mask[:, i]
                prev_mask = mask[:, i-1]
                # Only add score if both current and previous positions are not masked
                valid = prev_mask & mask_i
                
                # Add transition score
                prev_tag = tags[:, i-1]
                curr_tag = tags[:, i]
                transition_score = self.transitions[prev_tag, curr_tag]
                score += transition_score * valid
                
                # Add emission score
                emission_score = emissions[:, i].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
                score += emission_score * mask_i
            else:
                # Add transition score
                prev_tag = tags[:, i-1]
                curr_tag = tags[:, i]
                score += self.transitions[prev_tag, curr_tag]
                
                # Add emission score
                score += emissions[:, i].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
        
        # Add transition to end tag
        if mask is not None:
            # Find last unmasked position for each sequence
            seq_ends = mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(batch_size, device=emissions.device)
            last_tags = tags[batch_indices, seq_ends]
            score += self.end_transitions[last_tags]
        else:
            last_tags = tags[:, -1]
            score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_normalizer(self, emissions, mask):
        """Compute the partition function using the forward algorithm."""
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize alpha for the first step
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
        
        for i in range(1, seq_length):
            # Broadcast alpha to all possible next tags
            # shape: (batch_size, num_tags, 1)
            alpha_expanded = alpha.unsqueeze(2)  # (batch_size, num_tags, 1)
            
            # Broadcast transitions to all samples
            # shape: (1, num_tags, num_tags)
            transitions_expanded = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            
            # Add transition scores
            # shape: (batch_size, num_tags, num_tags)
            scores = alpha_expanded + transitions_expanded  # (batch_size, num_tags, num_tags)
            
            # Log-sum-exp over previous tags
            # shape: (batch_size, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch_size, num_tags)
            
            # Add emission scores for current step
            emissions_i = emissions[:, i]  # (batch_size, num_tags)
            new_alpha = new_alpha + emissions_i  # (batch_size, num_tags)
            
            if mask is not None:
                mask_i = mask[:, i].unsqueeze(1)  # (batch_size, 1)
                # If mask_i is False, keep previous alpha values
                alpha = torch.where(mask_i.bool(), new_alpha, alpha)
            else:
                alpha = new_alpha
        
        # Add transition to end tag
        alpha += self.end_transitions.unsqueeze(0)  # (batch_size, num_tags)
        
        # Log-sum-exp over all tags
        return torch.logsumexp(alpha, dim=1)  # (batch_size,)
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi decoding to find the most likely tag sequence."""
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize backpointers and viterbi variables
        backpointers = torch.zeros(batch_size, seq_length, num_tags, dtype=torch.long, device=emissions.device)
        
        # Initialize viterbi variables for first step
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
        
        for i in range(1, seq_length):
            # Broadcast viterbi to all possible next tags
            # shape: (batch_size, num_tags, 1)
            viterbi_expanded = viterbi.unsqueeze(2)  # (batch_size, num_tags, 1)
            
            # Broadcast transitions to all samples
            # shape: (1, num_tags, num_tags)
            transitions_expanded = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            
            # Add transition scores
            # shape: (batch_size, num_tags, num_tags)
            scores = viterbi_expanded + transitions_expanded  # (batch_size, num_tags, num_tags)
            
            # Find best previous tag for each current tag
            # shape: (batch_size, num_tags)
            best_scores, best_tags = torch.max(scores, dim=1)  # (batch_size, num_tags), (batch_size, num_tags)
            backpointers[:, i] = best_tags
            
            # Add emission scores
            emissions_i = emissions[:, i]  # (batch_size, num_tags)
            new_viterbi = best_scores + emissions_i  # (batch_size, num_tags)
            
            if mask is not None:
                mask_i = mask[:, i].unsqueeze(1)  # (batch_size, 1)
                viterbi = torch.where(mask_i.bool(), new_viterbi, viterbi)
            else:
                viterbi = new_viterbi
        
        # Add transition to end tag
        viterbi += self.end_transitions.unsqueeze(0)  # (batch_size, num_tags)
        
        # Find best final tag
        best_scores, best_tags = torch.max(viterbi, dim=1)  # (batch_size,), (batch_size,)
        
        # Backtrack to find best path
        best_paths = torch.zeros(batch_size, seq_length, dtype=torch.long, device=emissions.device)
        best_paths[:, -1] = best_tags
        
        for i in range(seq_length - 2, -1, -1):
            batch_indices = torch.arange(batch_size, device=emissions.device)
            best_tags = backpointers[batch_indices, i + 1, best_tags]
            best_paths[:, i] = best_tags
        
        return best_paths
