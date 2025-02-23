import torch 
import torch.nn.functional as F

class BeamSearchGenerator:
    """
    Beam search for autoregressive decoding.
    """
    def __init__(self,
                 model, 
                 config):
        self.model = model
        self.config = config
        self.beam_size = config.beam_size
        self.pad_token_id = config.pad_token_id
        self.begin_sequence_token_id = config.begin_sequence_token_id
        self.end_sequence_token_id = config.end_sequence_token_id
    
    @torch.no_grad()
    def generate(self,
                 input_ids,
                 attention_mask=None):
        """
        Generate sequences using beam search.
        Args: 
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Generated Token ids (batch_size, until end of sequence)
        """
        encoder_output = self.model.encoder(
            input_ids,
            attention_mask=attention_mask
        ) 
        batch_size = input_ids.shape[0]        
        
        # for all the sequences in the batch, initialize the beam search with the BOS token
        # this will be fed to the decoder as the starting state
        curr_ids = torch.full(
            (batch_size, 1),
            self.begin_sequence_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        
        # each seq in the batch will have a "beam size" number of scores each score correspond to the 
        # cumulative log probability of the sequence until the last generated token
        beam_scores = torch.zeros(
            (batch_size, self.beam_size),
            dtype=torch.float,
            device=input_ids.device
        )
        
        # so say that the encoder output is (batch_size, seq_len, d_model)
        # we create a new dim of beam size and repeat the encoder output beam_size times for each seq in the batch
        # it is like saying each seq has beam size number of decoding paths 
        encoder_outputs = encoder_output.unsqueeze(1).expand(
            batch_size,
            self.beam_size,
            -1,
            -1
        )
        
        done = [False for _ in range(batch_size)]
        
        while not all(done):
            # prevent infinite generation ( just in case )
            if curr_ids.shape[1] >= self.config.max_seq_len:
                break
            
            decoder_outputs = self.model.decoder(
                curr_ids,
                encoder_outputs=encoder_outputs,
                self_attention_mask=self._get_causal_mask(curr_ids)
            )
            
            # get the last token logits for each seq in the batch
            next_token_logits = decoder_outputs[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            vocab_size = next_token_scores.shape[-1]
            # so for each seq in the batch, we have beam_size number of scores 
            # we sort these scores in descending order and get the top k scores
            next_scores = next_token_scores + beam_scores.unsqueeze(-1)
            next_scores = next_scores.view(batch_size, -1) 
            next_tokens = torch.argsort(next_scores, dim=-1, descending=True)
            next_scores = torch.gather(next_scores, dim=-1, index=next_tokens)
            
            # we want to identify which beam each token belongs to, so we can continue with the beam that scored the highest
            # so we divide the token id by the vocabulary size
            # the quotient will be the beam id
            next_beams = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            # now we extract the actual token id from the flattened index
            next_tokens = next_tokens % vocab_size
            
            # keep only the top beam_size scores for each seq in the batch
            beam_scores = next_scores[:, :self.beam_size]
            
            # now we concatenate the current ids with the new tokens ( this will be fed to the decoder )
            curr_ids = torch.cat([curr_ids[next_beams], next_tokens.unsqueeze(-1)], dim=-1)
            
            eos_mask = next_tokens == self.end_sequence_token_id
            if eos_mask.any():
                for idx in range(batch_size):
                    if not done[idx] and eos_mask[idx].any():
                        done[idx] = True
        
        # return the top scoring beam
        return curr_ids[:, 0]
    
    def _get_causal_mask(self,
                         input_ids):
        """ Create a causal mask for decoder no peeking *|*"""
        batch_size, seq_len = input_ids.shape
        mask = torch.triu(
            torch.ones((seq_len, seq_len)), 
            dtype=torch.bool,
            diagonal=1
        ).to(input_ids.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask