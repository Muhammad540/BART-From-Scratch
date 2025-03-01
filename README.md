# BART
An Encoder-Decoder Transformer from scratch to summarize news articles

## External Libs used 
1. pytorch 
2. Huggingface transformers (only for tokenizer)
3. Evaluate (for rouge metrics)

## Training 

```bash
python lets_train.py --mode train --config engine/general_utils/model_config.yaml
```

## Evaluation 

```bash
python lets_train.py --mode evaluate --config engine/general_utils/model_config.yaml --checkpoint <path_to_checkpoint>
```

