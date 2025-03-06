import marimo

__generated_with = "0.11.13"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#Lets implement 'BART', a seq-seq transformer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        BART was originally introduced in the paper called, 'BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension' by by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

        For more information please refer to: [HuggingFace BART ](https://huggingface.co/docs/transformers/en/model_doc/bart#implementation-notes)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The goal of this project is not to beat any bechmark or create a state of the art performing model. The goal is to learn, write from scratch, implement the minor details that we dont get to interface with when using Api's or using pretrained models.

        I want to only use pytorch and in some places uses the huggingface libs (for tokenizer and evaluation). We will train the model on the objective of text summarization, and note that we will not be using any Pretrained weights, so the performance of our model wont be that good but that is not the goal as well.

        The training objective is that the model given an 'Article' and its 'Summary' should be trained to correctly output the Summary. We will use the CNN/Dailymail dataset provided by huggingface, you can read more about it here: [CNN/DailyMail Dataset HuggingFace](https://huggingface.co/datasets/abisee/cnn_dailymail)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Components to Implement:
        1. Loading the Dataset
        2. Defining the model config
        2. Token Embeddings
        3. Positional Encodings
        4. Multi-Head Attention
        5. Encoder Block
        6. Decoder Block
        7. Transformer
        8. Sampler (Beam Search)
        9. Training Pipeline
        10. Evaluation Script

        My inspiration:
        1. [Attention is All you Need](https://arxiv.org/abs/1706.03762)
        2. [BART](https://arxiv.org/abs/1910.13461)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### All the necessary imports""")
    return


app._unparsable_cell(
    r"""
    !pip install datasets
    !pip install transformers
    !pip install evaluate
    !pip install rouge_score
    """,
    name="_"
)


@app.cell
def _():
    from datasets import load_dataset
    import evaluate as evaluation_metrics
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import BartTokenizer
    from transformers import get_linear_schedule_with_warmup
    from transformers import PreTrainedModel, GenerationConfig, GenerationMixin
    from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
    import math
    from typing import Optional
    from tqdm import tqdm
    import os
    return (
        BartTokenizer,
        BaseModelOutput,
        DataLoader,
        Dataset,
        F,
        GenerationConfig,
        GenerationMixin,
        Optional,
        PreTrainedModel,
        Seq2SeqLMOutput,
        evaluation_metrics,
        get_linear_schedule_with_warmup,
        load_dataset,
        math,
        nn,
        optim,
        os,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### loading dataset utilities""")
    return


@app.cell
def _(BartTokenizer, DataLoader, Dataset, load_dataset, torch):
    class SummarizationDataset(Dataset):
        def __init__(self,
                     articles,
                     summaries,
                     tokenizer=BartTokenizer.from_pretrained("facebook/bart-base"),
                     max_length=512):
            self.articles = articles
            self.summaries = summaries
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.articles)

        def __getitem__(self, idx):
            article = self.articles[idx]
            summary = self.summaries[idx]

            # tokenize the article and summary
            article_encodings = self.tokenizer(article,
                                               max_length=self.max_length,
                                               padding="max_length",
                                               truncation=True)
            summary_encodings = self.tokenizer(summary,
                                               max_length=self.max_length,
                                               padding="max_length",
                                               truncation=True)

            # this masking is needed to distinguish between padding and actual tokens
            article_mask = article_encodings["attention_mask"]
            summary_mask = summary_encodings["attention_mask"]

            # perform teacher forcing
            # decoder input ids are what we feed into the decoder
            # labels is what we expect the decoder output to be
            ''' Breif summary of how teacher forcing works:
            In the decoder, we feed the input ids with the last predicted token removed
            The first token in the label is removed since it is the start token and we want to predict the next token
            So say the input sentence is "Hello, how are you?" with token ids: [<start> , 8 , 9 , 10 , 11 , 12 , <end>]
            The decoder input ids are: [<start> , 8 , 9 , 10 , 11 , 12]
            The labels are: [8 , 9 , 10 , 11 , 12 , <end>]
            So when '<start>' is fed into the decoder, the output should be 8, so the label is 8
            The next input to the decoder is 8, so the output should be 9, so the label is 9
            This continues until the last token is reached
            '''
            return {
                "input_ids": torch.tensor(article_encodings["input_ids"]),
                "attention_mask": torch.tensor(article_mask),
                "decoder_input_ids": torch.tensor(summary_encodings["input_ids"][:-1]),
                "decoder_attention_mask": torch.tensor(summary_mask[:-1]),
                "labels": torch.tensor(summary_encodings["input_ids"][1:])
            }


    def load_cnn_dailymail(batch_size,
                           max_length):
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        train_dataset = SummarizationDataset(
            articles=dataset["train"]["article"],
            summaries=dataset["train"]["highlights"],
            tokenizer=tokenizer,
            max_length=max_length)

        val_dataset = SummarizationDataset(
            articles=dataset["validation"]["article"],
            summaries=dataset["validation"]["highlights"],
            tokenizer=tokenizer,
            max_length=max_length)

        test_dataset = SummarizationDataset(
            articles=dataset["test"]["article"],
            summaries=dataset["test"]["highlights"],
            tokenizer=tokenizer,
            max_length=max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    return SummarizationDataset, load_cnn_dailymail


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### before going any furthur, lets also define some model configuration""")
    return


@app.cell
def _():
    model_config = {"model_name":"bart_seq2seq",
                    "vocab_size": 50265,               # defines the size of the token vocab
                    "max_seq_len": 512,                # max sequence of length that the model can process
                    "d_model": 512,                    # dimension of embedding and hidden states throughout the model
                    "encoder_layers": 12,              # number of encoder/decoder layers in the transformer
                    "decoder_layers": 12,              # number of encoder/decoder layers in the transformer
                    "encoder_attention_heads": 8,      # number of attention heads in multihead attention
                    "decoder_attention_heads": 8,      # number of attention heads in multihead attentino
                    "encoder_ff_dim": 1024,            # hidden dim size in FF network
                    "decoder_ff_dim": 1024,            # hidden dim size in FF network
                    "dropout": 0.1,                    # general dropout rate throughout the model
                    "pad_token_id": 1,                 # special token ids used in generation and processing
                    "begin_sequence_token_id": 0,      # special token ids used in generation and processing
                    "end_sequence_token_id": 2,        # special token ids used in generation and processing
                    "beam_size": 4,                    # control the num of paths in beam search
                    "batch_size": 16,                  # batch size (how many examples are batched together to train in parallel)
                    "learning_rate": 3e-5,             # learning rate decides how big of a step you should take in the GD
                    "warmup_steps": 500,               # used in the learning rate scheduler
                    "max_grad_norm": 1.0,              # maximum gradient norm used for clipping (used in the training loop  )
                    "num_epochs": 1,                   # full passes through the training data
                    "num_samples":500,                 # how many samples to generate in the evaluation (for vibe check)
                    "eval_sample":3,                   # num samples that we get in the evaluation during training
                    "log_every":1000,                  # defines logging frequency
                    "start_eval_gen":100}              # again used for logging frequency
    return (model_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### **a note on what is 'warm up steps' ?**

        *So when training neural networks, you can keep the learning rate constant, but it often leads to better, stable, faster convergence if you 'smartly' adjust the learning rate. You can use exponential decay learning rate, step decay, cosine annealing, etc. But i decided to use linear learning rate scheduler with warmup.*

        *warmup basically works in this way, you start with a very small value and gradually increase to a predefined maximum value over a certain number of steps (which is defined by the warmup steps). Once the warmup is complete, the learning rate follows some decay schedule (like linear in this case but can be exponential as well).* *Read this [they have some nice visuals]*(https://docs.anyscale.com/llms/finetuning/guides/modify_hyperparams/)


        #### **a note on 'max grad norm' ?**

        *gradient clipping is used in our training loop to avoid exploding gradients*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### embeddings  (this includes both the token embeddings and positional embedding)""")
    return


@app.cell
def _(math, nn, torch):
    class PositionalEncoding(nn.Module):
        """
        Implements sinusoidal positional encoding as described in "Attention is All You Need"

        This will add positional information to the input embeddings since transformer architectures
        don't inherently understand the sequence order
        """
        def __init__(self,
                     embedding_dim: int,
                     max_sequence_length: int,
                     dropout_prob: float):
            super().__init__()

            self.dropout = nn.Dropout(p=dropout_prob)

            position = torch.arange(max_sequence_length).unsqueeze(1)
            # implement the sinusoidal encoding (formula at page 6 of the paper)
            denominator = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

            pe = torch.zeros(1, max_sequence_length, embedding_dim)
            pe[0,:,0::2] = torch.sin(position * denominator)
            pe[0,:,1::2] = torch.cos(position * denominator)

            # since the positional encoding are not learned, we can simply store them in model state
            self.register_buffer('pe', pe)

        def forward(self,
                    x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input Tensor of shape [batch size, seq len, embedding dim]
            Returns:
                Tensor with positional encoding added
            """
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class TokenEmbedding(nn.Module):
        """
        Embeds the input tokens into a vector space of dimension d_model
        Also we have to scale the embeddings by sqrt(d_model) as described in the paper page 5
        """
        def __init__(self,
                     vocab_size: int,
                     embedding_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_dim = embedding_dim

        def forward(self,
                    x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input Tensor of shape [batch size, seq len]

            Returns:
                Embedded tokens scaled by sqrt(d_model) [batch size, seq len, embedding dim]
            """
            return self.embedding(x) * math.sqrt(self.embedding_dim)
    return PositionalEncoding, TokenEmbedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### code up the multihead attention""")
    return


@app.cell
def _(F, Optional, math, nn, torch):
    class MultiHeadAttention(nn.Module):
        """
        Multi-head attention mechanism similar to what is described in 'Attention is all you need'
        You linearly project each of the queries, keys and values with different learned projections
        And in each head, self attention is applied, the result are concatenated and projected to the output

        This basically helps the model to jointly attend to information from different positions, with different represenatational subspaces
        For example, if we have a sentence "He went to the bank to get some money, and later went for a walk along the river bank"
        The model should be able to attend to the words "bank" in different ways:
        - "bank" as a place to get money
        - "bank" as a place to walk along the river
        """
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            # dk is the dimension of each head's key, query and value

            # linear layers for queries, keys and values projections
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

            # e.g 8 heads * 64 dk -> 512 d model
            self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            The popular attention mechanism

            Args:
                query: [batch size, num heads, seq len q, d_k]
                key: [batch size, num heads, seq len k, d_k]
                value: [batch size, num heads, seq len v, d_k]
                mask: [batch size, 1, seq len q, seq len k]
            Returns:
                Tensor of shape [batch size, num heads, seq len, d_k]
            """
            similarity = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                similarity = similarity.masked_fill(mask == 0, -1e9)
            attention_weights = F.softmax(similarity, dim=-1)
            attention_weights = self.dropout(attention_weights)

            attention_scores = torch.matmul(attention_weights, v)
            return attention_scores, attention_weights

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                query: [batch size, seq len q, d model]
                key: [batch size, seq len k, d model]
                value: [batch size, seq len v, d model]
                mask: [batch size, 1, seq len q, seq len k]

            Returns:
                Two tensors output and attention weights
                output: [batch size, num heads, seq len q, d model]
                attention weights: [batch size, num heads, seq len q, seq len k]
            """
            batch_size = query.size(0)

            # view each of q,k,v to [batch size, seq len, num heads, dk]
            # so if (32, 10, 512) -> (32, 10, 8, 64)
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # chunk it for each head
            # why transpose?
            # [batch size, seq len, num heads, dk] -> [batch size, num heads, seq len, dk]
            q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            attention_scores, attention_weights = self.attention(q, k, v, mask)

            # combine the heads (sounds kinda weird)
            # lets break down the math
            # output -> [batch size, num heads, seq len, dk]
            # revoke the transpose -> [batch size, seq len, num heads, dk]
            # view it back to [batch size, seq len, num heads * dk]
            combined = attention_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

            # project it back to d_model
            projected = self.out_proj(combined)

            return projected, attention_weights
    return (MultiHeadAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### code up the 'Encoder Block'""")
    return


@app.cell
def _(MultiHeadAttention, nn):
    class EncoderBlock(nn.Module):
        """
        A simple/single encoder block:
        - Multi-head attention mechanism
        - Feed-forward neural network
        - Layer normalization
        NOTE: GELU instead of ReLU is used as per the BART paper
        """
        def __init__(self,
                     config):
            super().__init__()
            self.d_model = config["d_model"]
            self.encoder_attention_heads = config["encoder_attention_heads"]
            self.dropout = config["dropout"]
            self.encoder_ff_dim = config["encoder_ff_dim"]

            self.self_attention = MultiHeadAttention(
                d_model=self.d_model,
                num_heads=self.encoder_attention_heads,
                dropout=self.dropout,
            )

            self.layer_norm1 = nn.LayerNorm(self.d_model)
            self.layer_norm2 = nn.LayerNorm(self.d_model)

            self.feed_forward = nn.Sequential(
                nn.Linear(self.d_model, self.encoder_ff_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.encoder_ff_dim, self.d_model),
                nn.Dropout(self.dropout)
            )
        def forward(self,
                    x,
                    attention_mask=None):
            # tidbit: residual connection is used to avoid the vanishing gradient problem
            # also the way we have implemented this is called "pre layer normalization" since BART paper uses this
            residual = x

            # attention with residual connection
            x = self.layer_norm1(x)

            if attention_mask is not None:
                # convert from (batch size, seq len) to (batch size, 1, 1, seq len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # now expand to (batch size, 1, seq len, seq len)
                seq_len = attention_mask.size(3)
                attention_mask = attention_mask.expand(-1, -1, seq_len, -1)

            x, _ = self.self_attention(
                query=x,
                key=x,
                value=x,
                mask=attention_mask
            )

            x = x + residual
            # feedforward with residual connection
            residual = x
            x = self.layer_norm2(x)
            x = self.feed_forward(x)
            x = x + residual
            return x
    return (EncoderBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### code up the 'Decoder Block'""")
    return


@app.cell
def _(MultiHeadAttention, nn):
    class DecoderBlock(nn.Module):
        """
        A single decoder block for BART:
        - Masked multi-head attention (No peeking)
        - Multi-head cross attention with encoder outputs
        - Feed forward NN
        - layer Normalization
        - Residual connections
        NOTE: Uses Pre-LN arch like the encoder also uses GELU
        """
        def __init__(self,
                     config):
            super().__init__()
            self.d_model = config["d_model"]
            self.decoder_attention_heads = config["decoder_attention_heads"]
            self.decoder_ff_dim = config["decoder_ff_dim"]
            self.dropout = config["dropout"]


            self.masked_self_attention = MultiHeadAttention(
                d_model=self.d_model,
                num_heads=self.decoder_attention_heads,
                dropout=self.dropout,
            )

            self.cross_attention = MultiHeadAttention(
                d_model=self.d_model,
                num_heads=self.decoder_attention_heads,
                dropout=self.dropout,
            )

            self.layer_norm1 = nn.LayerNorm(self.d_model)
            self.layer_norm2 = nn.LayerNorm(self.d_model)
            self.layer_norm3 = nn.LayerNorm(self.d_model)

            self.feed_forward = nn.Sequential(
                nn.Linear(self.d_model, self.decoder_ff_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.decoder_ff_dim, self.d_model),
                nn.Dropout(self.dropout)
            )
        def forward(self,
                    x,
                    encoder_output,
                    self_attention_mask=None,
                    cross_attention_mask=None):
            """
            we need both the self attention and cross attention mask,
            since the decoder uses its Query embeddings to extract information from the
            Encoders output and the cross attention mask is needed to prevent the decoder
            from attending to 'padding tokens' in the encoder outputs. While the self attention mask
            is 'causal' in nature. We ensure that the decoder doesnot peek into the future tokens.
            Args:
                encoder_output: (batch_size, seq_len, d_model)
                self_attention_mask: (batch_size, seq_len, seq_len)
                cross_attention_mask: (batch_size, seq_len, seq_len)
            Returns:
                (batch_size, seq_len, d_model)
            """
            residual = x
            x = self.layer_norm1(x)
            x, _ = self.masked_self_attention(
                query=x,
                key=x,
                value=x,
                mask=self_attention_mask
            )
            x = x + residual

            # important thing to notice here is that the:
            # query is coming from the decoder itself
            # key and value are coming from the encoder
            # so basically decoder is looking at itself and attending to the encoder outputs
            # this is why it is called "cross attention"
            residual = x
            x = self.layer_norm2(x)
            x, _ = self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask = cross_attention_mask
            )
            x = x + residual

            residual = x
            x = self.layer_norm3(x)
            x = self.feed_forward(x)
            x = x + residual

            return x
    return (DecoderBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### transformer (~ it is basically collection of encoder and decoder blocks stacked up)""")
    return


@app.cell
def _(DecoderBlock, EncoderBlock, PositionalEncoding, TokenEmbedding, nn):
    class BartEncoder(nn.Module):
        """
        Fully stacked with encoder blocks
        A single encoder block is defined in encoder.py
        """
        def __init__(self,
                     config):
            super().__init__()

            self.d_model = config["d_model"]
            self.vocab_size = config["vocab_size"]
            self.max_seq_len = config["max_seq_len"]
            self.encoder_layers = config["encoder_layers"]
            self.dropout = config["dropout"]

            self.embedding = TokenEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.d_model
            )

            self.positional_encoding = PositionalEncoding(
                embedding_dim=self.d_model,
                max_sequence_length=self.max_seq_len,
                dropout_prob=self.dropout
            )

            # stack the encoder blocks
            self.encoder_stacked_layers = nn.ModuleList([
                EncoderBlock(config) for _ in range(self.encoder_layers)
            ])

        def forward(self,
                    x,
                    attention_mask=None):
            """
            Args:
                x: (batch_size, seq_len)
                attention_mask: (batch_size, seq_len)
            Returns:
                (batch_size, seq_len, d_model)
            """
            x = self.embedding(x)
            x = self.positional_encoding(x)

            for layer in self.encoder_stacked_layers:
                x = layer(x, attention_mask)

            return x

    class BartDecoder(nn.Module):
        """
        Fully stacked with decoder blocks
        A single decoder block is defined in decoder.py
        """
        def __init__(self,
                     config):
            super().__init__()
            self.d_model = config["d_model"]
            self.vocab_size = config["vocab_size"]
            self.max_seq_len = config["max_seq_len"]
            self.decoder_layers = config["decoder_layers"]
            self.dropout = config["dropout"]

            self.embedding = TokenEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.d_model
            )

            self.positional_encoding = PositionalEncoding(
                embedding_dim=self.d_model,
                max_sequence_length=self.max_seq_len,
                dropout_prob=self.dropout
            )

            self.decoder_stacked_layers = nn.ModuleList([
                DecoderBlock(config) for _ in range(self.decoder_layers)
            ])

            self.final_layer_norm = nn.LayerNorm(self.d_model)
            self.final_dropout = nn.Dropout(self.dropout)
            self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        def forward(self,
                    x,
                    encoder_output,
                    self_attention_mask=None,
                    cross_attention_mask=None):
            """
            Args:
                x: (batch_size, seq_len)
                encoder_output: (batch_size, seq_len, d_model)
                self_attention_mask: (batch_size, seq_len, seq_len)
                cross_attention_mask: (batch_size, seq_len, seq_len)
            Returns:
                (batch_size, seq_len, d_model)
            """
            x = self.embedding(x)
            x = self.positional_encoding(x)

            for layer in self.decoder_stacked_layers:
                x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)

            x = self.final_layer_norm(x)
            x = self.final_dropout(x)
            x = self.output_projection(x)

            return x
    return BartDecoder, BartEncoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### sampler, two imples
        1. Greedy Gen
        2. Beam Search
        """
    )
    return


@app.cell
def _(torch):
    class GreedyGenerator:
        """
        Simple greedy decoding for autoregressive generation.
        """
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.max_seq_len = config["max_seq_len"]
            self.begin_sequence_token_id = config["begin_sequence_token_id"]
            self.end_sequence_token_id = config["end_sequence_token_id"]

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
            done = [False for _ in range(batch_size)]

            # generate tokens one by one until EOS or max seq len limit
            step = 0
            while not all(done) and step < self.max_seq_len -1 :
                step += 1
                curr_len = curr_ids.size(1)

                # causal mask for decoder
                causal_mask = self._get_causal_mask(curr_ids)
                # cross-attention mask
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

                for i in range(batch_size):
                    if next_token[i, 0] == self.end_sequence_token_id:
                        done[i] = True

                if all(done):
                    print(f"All sequences completed at step {step}")

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
    return (GreedyGenerator,)


@app.cell
def _(F, torch):
    class BeamSearchGenerator:
        """
        Beam search for autoregressive decoding.
        """
        def __init__(self,
                     model,
                     config):
            self.model = model
            self.beam_size = config["beam_size"]
            self.max_seq_len = config["max_seq_len"]
            self.pad_token_id = config["pad_token_id"]
            self.begin_sequence_token_id = config["begin_sequence_token_id"]
            self.end_sequence_token_id = config["end_sequence_token_id"]

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
            """causal mask for decoder no peeking """
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
    return (BeamSearchGenerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### engine: combines the BART model in one place""")
    return


@app.cell
def _(BartDecoder, BartEncoder, BeamSearchGenerator, F, GreedyGenerator, nn):
    class BART(nn.Module):
        """
        BART: Denoising Seq-to-Seq Pre-training for
        Natural Language Generation, Translation, and Comprehension
        """
        def __init__(self,
                     config):
            super().__init__()

            self.config = config
            self.max_seq_len = config["max_seq_len"]
            self.encoder = BartEncoder(config)
            self.decoder = BartDecoder(config)
            # Tie the decoder embedding and output projection layer weights
            # Reason ?
            # 1. Reduce model params
            # 2. Act as a regularizer
            # 3. Since it is just an inverse operation (embedding: input tokens -> embedding, projection: embedding -> output tokens)
            self.decoder.output_projection.weight = self.decoder.embedding.embedding.weight

        def forward(self,
                    input_ids,
                    decoder_input_ids,
                    encoder_padding_mask=None,
                    encoder_decoder_attention_mask=None,
                    decoder_causal_mask=None,
                    labels=None):
            """
            Args:
                input_ids: (batch_size, seq_len)
                decoder_input_ids: (batch_size, seq_len)
                encoder_padding_mask: (batch_size, seq_len)
                encoder_decoder_attention_mask: (batch_size, seq_len, seq_len)
                decoder_causal_mask: (batch_size, seq_len, seq_len)
                labels: (batch_size, seq_len)
            Returns:
                (outputs, loss) if labels are provided else (output tokens)
                outputs: (batch_size, seq_len, vocab_size)
            """
            encoder_output = self.encoder(input_ids,
                                          attention_mask=encoder_padding_mask)
            decoder_output = self.decoder(decoder_input_ids,
                                          encoder_output=encoder_output,
                                          self_attention_mask=decoder_causal_mask,
                                          cross_attention_mask=encoder_decoder_attention_mask)

            if labels is not None:
                loss = F.cross_entropy(
                    # (batch_size * seq_len, vocab_size)
                    decoder_output.view(-1, decoder_output.size(-1)),
                    # (batch_size * seq_len)
                    labels.view(-1),
                    ignore_index=-100
                )
                return decoder_output, loss

            return decoder_output

        def generate(self,
                     input_ids,
                     use_greedy,
                     attention_mask=None):
            '''
            This method will be used to generate summaries using the beam search algo that we implemented

            ArgS:
                input_ids: input token ids (batch size, seq len)
                attention_mask: Attention mask for the input (batch size, seq len)
            '''
            if use_greedy:
                greedy_generator = GreedyGenerator(self, self.config)
                return greedy_generator.generate(
                    input_ids,
                    attention_mask=attention_mask
                )
            else:
                beam_generator = BeamSearchGenerator(self, self.config)
                return beam_generator.generate(input_ids, attention_mask)
    return (BART,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### training utilities""")
    return


@app.cell
def _(evaluation_metrics):
    def calculate_rouge(predictions, references):
        rouge = evaluation_metrics.load("rouge")

        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        return {k: round(v,4) for k,v in results.items()}
    return (calculate_rouge,)


@app.cell
def _(
    BART,
    BartTokenizer,
    calculate_rouge,
    get_linear_schedule_with_warmup,
    load_cnn_dailymail,
    math,
    model_config,
    optim,
    os,
    torch,
    tqdm,
):
    def train(save_dir="checkpoints"):
        """
        train bart model on the cnn/dailymail dataset

        Args:
            config_path: model configuration yaml file
            batch_size: batch size for training
            num_epochs: number of training epochs
            learning_rate: learning rate for the optimizer
            warmup_steps: number of warmup steps for the learning rate scheduler
            max_grad_norm: maximum gradient norm for gradient clipping
            save_dir: directory to save model checkpoints
        """
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        os.makedirs(save_dir, exist_ok=True)
        batch_size = model_config["batch_size"]
        max_seq_len = model_config["max_seq_len"]
        num_epochs = model_config["num_epochs"]
        learning_rate = model_config["learning_rate"]
        warmup_steps = model_config["warmup_steps"]
        max_grad_norm = model_config["max_grad_norm"]
        log_every = model_config["log_every"]
        eval_sample = model_config["eval_sample"]
        start_eval_gen = model_config["start_eval_gen"]

        train_dataloader, val_dataloader, test_dataloader = load_cnn_dailymail(batch_size=batch_size,max_length=max_seq_len)
        model = BART(model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")
        model.to(device)

        optimizer=optim.AdamW(model.parameters(), lr=learning_rate)

        total_steps = len(train_dataloader)*num_epochs

        scheduler=get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # we will keep track of imp metrics and also during the training process
        # we will generate summaries (for vibe checks) and to calculate rouge scores
        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "perplexity": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "learning_rate": []
        }

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            epoch_step = 0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(device)
                labels = batch["labels"].to(device)

                seq_len = decoder_input_ids.size(1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

                # create a cross attention mask with dim [batch_size, 1, decoder_seq_len, encoder_seq_len]
                # this will allow the decoder position to attend to all encoder positions
                # Note: we use the encoder's attention mask, not decoder attention mask because
                # we need to prevent attending to padding tokens in the encoder sequence
                encoder_decoder_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1,-1,seq_len,-1)

                # forward pass with causal mask
                _,loss = model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_padding_mask=attention_mask,
                    encoder_decoder_attention_mask=encoder_decoder_attention_mask,
                    decoder_causal_mask=causal_mask,
                    labels=labels
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()

                # compute perplexity = exp(loss)
                perplexity = math.exp(loss.item())
                total_loss += loss.item()
                epoch_step += 1
                global_step += 1

                progress_bar.set_postfix(
                    {
                        "step": global_step,
                        "loss": loss.item(),
                        "avg_loss": total_loss/(epoch_step + 1),
                        "ppl": perplexity,
                        "lr": scheduler.get_last_lr()[0]
                    }
                )

                if global_step % log_every == 0:
                    metrics_history["train_loss"].append((global_step, loss.item()))
                    metrics_history["perplexity"].append((global_step, perplexity))
                    metrics_history["learning_rate"].append((global_step, scheduler.get_last_lr()[0]))

                    # after waiting for start_eval_gen steps, we will generate some summaries
                    if global_step > start_eval_gen:
                        print("\n---Sample Generation---")
                        sample_input = input_ids[0:1]
                        sample_mask = attention_mask[0:1]

                        generated_ids = model.generate(
                            input_ids=sample_input,
                            attention_mask=sample_mask,
                            use_greedy=True
                        )

                        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        original_text = tokenizer.batch_decode(sample_input, skip_special_tokens=True)

                        print(f"Original: {original_text[0]}")
                        print(f"Generated: {generated_text[0]}")

            model.eval()
            val_loss = 0
            val_step = 0
            all_preds = []
            all_targets = []

            # lets do some eval
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    decoder_input_ids = batch["decoder_input_ids"].to(device)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    seq_len = decoder_input_ids.size(1)
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

                    # create a cross attention mask with dim [batch_size, 1, decoder_seq_len, encoder_seq_len]
                    # this will allow the decoder position to attend to all encoder positions
                    encoder_seq_len = input_ids.size(1)
                    encoder_decoder_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1,-1,seq_len,-1)

                    # forward pass
                    _,loss = model(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        encoder_padding_mask=attention_mask,
                        encoder_decoder_attention_mask=encoder_decoder_attention_mask,
                        decoder_causal_mask=causal_mask,
                        labels=labels
                    )

                    val_loss += loss.item()
                    val_step += 1

                    if val_step <= eval_sample:
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_greedy=True
                        )

                        generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        reference_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
                        all_preds.extend(generated_summaries)
                        all_targets.extend(reference_summaries)

            avg_val_loss = val_loss / val_step
            val_perplexity = math.exp(avg_val_loss)
            if all_preds:
                rouge_scores = calculate_rouge(all_preds, all_targets)
    #            print(rouge_scores)
                rouge1 = rouge_scores['rouge1']
                rouge2 = rouge_scores['rouge2']
                rougel = rouge_scores['rougeL']

                metrics_history['rouge1'].append((global_step, rouge1))
                metrics_history['rouge2'].append((global_step, rouge2))
                metrics_history['rougeL'].append((global_step, rougel))

                print(f"\nROUGE Scores - R1: {rouge1:.4f}, R2: {rouge2:.4f}, RL: {rougel:.4f}")

            metrics_history['val_loss'].append((global_step, avg_val_loss))

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {total_loss/len(train_dataloader):.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Perplexity: {val_perplexity:.4f}")

            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': total_loss/len(train_dataloader),
                'val_loss': avg_val_loss,
                'metrics': metrics_history
            }, checkpoint_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': total_loss/len(train_dataloader),
                    'val_loss': avg_val_loss,
                    'metrics': metrics_history
                }, best_model_path)
                print(f"New best model saved with val_loss: {avg_val_loss:.4f}")

            print(f"Model saved to {checkpoint_path}")

            if all_preds:
                print("\nSample Generations:")
                for i in range(min(100, len(all_preds))):
                    print(f"\nGenerated: {all_preds[i]}...")
                    print(f"Reference: {all_targets[i]}...")

        print("Training complete")

        import json
        metrics_path = os.path.join(save_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f)
        print(f"Training metrics saved to {metrics_path}")
    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### evaluation metric (rouge score)
        [click to watch this video for better understanding on how the rouge score is calculated](https://www.youtube.com/watch?v=TMshhnrEXlg&ab_channel=HuggingFace)
        """
    )
    return


@app.cell
def _(
    BART,
    BartTokenizer,
    calculate_rouge,
    load_cnn_dailymail,
    model_config,
    torch,
    tqdm,
):
    def evaluate(checkpoint_path):
        '''
        evaluate the bart model on the cnn/dailymail dataset

        Args:
            checkpoint_path: path to the checkpoint file
            config_path: path to the config file
            batch_size: batch size for evaluation
            num_samples: number of samples to evaluate on
        '''
        batch_size = model_config["batch_size"]
        max_seq_len = model_config["max_seq_len"]
        num_samples = model_config["num_samples"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _,_,test_dataloader = load_cnn_dailymail(batch_size=batch_size, max_length=max_seq_len)

        model = BART(model_config)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        model.eval()

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        all_preds, all_targets, all_articles = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader, desc="Generating Summaries")):
                if i >= num_samples:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_greedy=True
                )

                generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                reference_summaries = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                articles = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                all_preds.extend(generated_summaries)
                all_targets.extend(reference_summaries)
                all_articles.extend(articles)

        rouge_scores = calculate_rouge(all_preds, all_targets)

        print(f"ROUGE Scores: ")
        print(f"ROUGE-1: {rouge_scores['rouge1']}")
        print(f"ROUGE-2: {rouge_scores['rouge2']}")
        print(f"ROUGE-L: {rouge_scores['rougeL']}")

        print("\n Sample Summaries:")
        for i in range(min(3, len(all_preds))):
            print(f"\nArticle {i+1}:")
            print(all_articles[i])
            print(f"\nGenerated Summary: {all_preds[i]}")
            print(f"Reference Summary: {all_targets[i]}")
    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### lets train""")
    return


@app.cell
def _(evaluate, train):
    def main():
        train(save_dir="checkpoints")
        checkpoint_path = "/content/checkpoints/best_model.pth"
        evaluate(checkpoint_path=checkpoint_path)

    if __name__ == "__main__":
        main()
    return (main,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
