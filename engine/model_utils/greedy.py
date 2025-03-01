import torch

class GreedyGenerator:
    """
    greedy decoding for autoregressive generation.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.begin_sequence_token_id = config.begin_sequence_token_id
        self.end_sequence_token_id = config.end_sequence_token_id
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None):
        """
        generate sequences using greedy decoding.
        Args: 
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            generated token ids (batch_size, until end of sequence)
        """
        # encode input
        encoder_output = self.model.encoder(
            input_ids,
            attention_mask=attention_mask
        )
        
        batch_size = input_ids.shape[0]
        enc_seq_len = encoder_output.size(1)
        
        # start with BOS token
        curr_ids = torch.full(
            (batch_size, 1),
            self.begin_sequence_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        

        # keep track of which sequences are done
        done = [False for _ in range(batch_size)]
        
        # generate tokens one by one
        for step in range(self.max_seq_len):
            if all(done):
                print(f"All sequences completed at step {step}")
                break
                
            curr_len = curr_ids.size(1)
            
            # create causal mask for decoder
            causal_mask = self._get_causal_mask(curr_ids)
            
            # create cross-attention mask
            if attention_mask is not None:
                cross_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, curr_len, -1)
            else:
                cross_mask = None
            
            # run the decoder
            decoder_outputs = self.model.decoder(
                curr_ids,
                encoder_output=encoder_output,
                self_attention_mask=causal_mask,
                cross_attention_mask=cross_mask
            )
            
            # get logits for next token
            next_token_logits = decoder_outputs[:, -1, :]
            
            # get the most likely token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # check if any sequence has reached the end of the sequence
            for i in range(batch_size):
                if next_token[i, 0] == self.end_sequence_token_id:
                    done[i] = True
        
        # remove BOS token if it's at the beginning
        final_outputs = []
        for i in range(batch_size):
            seq = curr_ids[i]
            if seq[0] == self.begin_sequence_token_id:
                seq = seq[1:]
            
            # make sure we have at least some tokens
            if len(seq) == 0:
                seq = torch.tensor([0], device=seq.device)
            
            final_outputs.append(seq)
        
        return torch.stack(final_outputs)
    
    def _get_causal_mask(self, input_ids):
        """a causal mask for decoder no peeking"""
        batch_size, seq_len = input_ids.shape
        
        # a square mask where the upper triangle is True (will be masked)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool),
            diagonal=1
        ).to(input_ids.device)
        
        # mask with dimensions [batch_size, 1, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
        
        # convert from bool mask to 0/1 mask as expected by the attention
        mask = ~mask  # invert since triu gives the part to mask out
        
        return mask
