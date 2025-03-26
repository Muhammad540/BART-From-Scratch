# BART
An Encoder-Decoder Transformer trained from scratch to summarize news articles using the CNN/DailyMail dataset purely for learning! 

## External Libs used 
1. pytorch 
2. Huggingface transformers (only for tokenizer)
3. Huggingface Evaluate (for rouge metrics)

## Training + Evaluation on colab (preferred)
To quickly start training and play with the model configuration, head to Notebook/TransformerFromScratch.ipynb.

Run the notebook on Google Colab. You will need to adjust the model configuration, such as the number of encoder and decoder layers, the number of attention heads, and the batch size, to avoid running out of GPU RAM if you are on a free trial. The cool part is that once you start training the model, you will observe the model's output as it begins learning and improving its generations!

Everything is configurable. If you're not GPU poor, just max out the config and watch the model's generations improve quite a bit.


## Training Locally

```bash
python lets_train.py --mode train --config ../general_utils/model_config.yaml
```

### Evaluation 

```bash
python lets_train.py --mode evaluate --config ../general_utils/model_config.yaml --checkpoint <path_to_checkpoint>
```

