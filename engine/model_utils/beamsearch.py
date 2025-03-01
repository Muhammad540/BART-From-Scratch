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
        self.max_seq_len = config.max_seq_len
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
        enc_seq_len = encoder_output.size(1)
        
        curr_ids = torch.full(
            (batch_size, 1),
            self.begin_sequence_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        
        beam_scores = torch.zeros(
            (batch_size, self.beam_size),
            dtype=torch.float,
            device=input_ids.device
        )
        
        # for beam search, expand each sequence in batch to beam_size copies
        # [batch_size, seq_len, d_model] -> [batch_size, beam_size, seq_len, d_model]
        # use repeat instead of expand to ensure contiguous memory
        encoder_outputs = encoder_output.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        
        # cross attention mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, beam_size, 1, seq_len]
            cross_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.beam_size, 1, 1)
        else:
            # create a default mask that allows attending to all encoder positions (in case we don't have a mask)
            cross_mask = torch.ones(
                (input_ids.size(0), self.beam_size, 1, encoder_output.size(1)),
                device=input_ids.device
            )
        
        # keep track of which sequences are done for each batch and beam
        done_beams = [[False for _ in range(self.beam_size)] for _ in range(batch_size)]
        
        # shape: [batch_size * beam_size, 1]
        curr_ids = curr_ids.repeat(self.beam_size, 1)  
        
        step = 0
        while not all(all(done) for done in done_beams) and step < self.max_seq_len:
            step += 1
            
            # reshape encoder outputs to match batch_size * beam_size
            flat_encoder_outputs = encoder_outputs.reshape(
                batch_size * self.beam_size, 
                enc_seq_len, 
                -1
            )
            
            if cross_mask is not None:
                # [batch_size, beam_size, 1, enc_seq_len] -> [batch_size, beam_size, curr_seq_len, enc_seq_len]
                curr_cross_mask = cross_mask.expand(-1, -1, curr_ids.size(1), -1)
                # reshape to [batch_size * beam_size, 1, curr_seq_len, enc_seq_len]
                curr_cross_mask = curr_cross_mask.reshape(
                    batch_size * self.beam_size, 
                    1, 
                    curr_ids.size(1), 
                    enc_seq_len
                )
            else:
                curr_cross_mask = None
            
            # run the decoder
            decoder_outputs = self.model.decoder(
                curr_ids,
                encoder_output=flat_encoder_outputs,
                self_attention_mask=self._get_causal_mask(curr_ids),
                cross_attention_mask=curr_cross_mask
            )
            
            # get logits for next token
            next_token_logits = decoder_outputs[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # reshape scores for beam search
            next_token_scores = next_token_scores.reshape(batch_size, self.beam_size, -1)
            
            # add current beam scores
            next_scores = beam_scores.unsqueeze(-1) + next_token_scores
            next_scores = next_scores.reshape(batch_size, -1)
            
            # select top-k scores and their indices
            topk_scores, topk_indices = next_scores.topk(self.beam_size, dim=1)
            
            # extract beam indices and token indices
            beam_indices = topk_indices // next_token_scores.size(-1)
            token_indices = topk_indices % next_token_scores.size(-1)
            beam_scores = topk_scores
            
            # prepare next iteration's token ids
            next_ids = []
            for batch_idx in range(batch_size):
                batch_next_ids = []
                
                for beam_idx in range(self.beam_size):
                    # skip if this beam is already done
                    if done_beams[batch_idx][beam_idx]:
                        # just copy the existing sequence
                        curr_beam_idx = beam_indices[batch_idx, beam_idx]
                        curr_seq = curr_ids[batch_idx * self.beam_size + curr_beam_idx].clone()
                        batch_next_ids.append(curr_seq)
                        continue
                    
                    curr_beam_idx = beam_indices[batch_idx, beam_idx]
                    # get current sequence for this beam
                    curr_seq = curr_ids[batch_idx * self.beam_size + curr_beam_idx].clone()
                    # append the next token
                    new_token = token_indices[batch_idx, beam_idx].unsqueeze(0)
                    next_seq = torch.cat([curr_seq, new_token], dim=0)
                    batch_next_ids.append(next_seq)
                    
                    # see if this beam generated EOS
                    if new_token.item() == self.end_sequence_token_id:
                        done_beams[batch_idx][beam_idx] = True
                
                # common bug: make sure all sequences in batch_next_ids have the same length before stacking
                max_len = max([seq.size(0) for seq in batch_next_ids])
                padded_batch_next_ids = []

                for seq in batch_next_ids:
                    if seq.size(0) < max_len:
                        padding = torch.full(
                            (max_len - seq.size(0),),
                            self.pad_token_id,
                            dtype=torch.long,
                            device=seq.device
                        )
                        padded_seq = torch.cat([seq, padding], dim=0)
                        padded_batch_next_ids.append(padded_seq)
                    else:
                        padded_batch_next_ids.append(seq)

                next_ids.append(torch.stack(padded_batch_next_ids))
            
            # stack and reshape token ids for next iteration
            next_ids = torch.stack(next_ids)  # [batch_size, beam_size, seq_len]
            curr_ids = next_ids.reshape(batch_size * self.beam_size, -1)
        
        # return the top beam for each sequence in batch
        final_outputs = []
        for batch_idx in range(batch_size):
            # get the beam with highest score
            best_beam_idx = torch.argmax(beam_scores[batch_idx])
            best_seq = curr_ids[batch_idx * self.beam_size + best_beam_idx]
            
            # when presenting the output, we don't want to include the BOS token
            if best_seq[0] == self.begin_sequence_token_id:
                best_seq = best_seq[1:]
            
            # when presenting the output, we don't want to include the EOS token
            if self.end_sequence_token_id in best_seq:
                # find the first occurrence of EOS and truncate
                eos_idx = (best_seq == self.end_sequence_token_id).nonzero(as_tuple=True)[0]
                if len(eos_idx) > 0:
                    best_seq = best_seq[:eos_idx[0]]
            
            if len(best_seq) == 0:
                best_seq = torch.tensor([0], device=best_seq.device)
            
            final_outputs.append(best_seq)
        
        return torch.stack(final_outputs)
    
    def _get_causal_mask(self, input_ids):
        """A causal mask for decoder no peeking """
        batch_size, seq_len = input_ids.shape
        
        # a square mask where the upper triangle is True (will be masked)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool),
            diagonal=1
        ).to(input_ids.device)
        
        # mask with dimensions [batch_size, 1, seq_len, seq_len]
        # the '1' dimension corresponds to the attention heads
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
        
        # convert from bool mask to 0/1 mask as expected by the attention
        # 0 means masked positions (don't attend here), 1 means valid positions
        # we have to invert the mask because the triu function gives the part to mask out
        mask = ~mask  
        
        return mask