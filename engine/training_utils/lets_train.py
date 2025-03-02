import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from engine.training_utils.train import train
from engine.training_utils.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BART model")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", help="Mode: train or evaluate")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "../general_utils/model_config.yaml"), help="Path to the model configuration file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (for evaluation)")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            config_path=args.config,
            save_dir=args.save_dir
        )
    elif args.mode == "evaluate":
        if not args.checkpoint:
            checkpoints = [f for f in os.listdir(args.save_dir) if f.endswith(".pt")]
            if not checkpoints:
                print("No checkpoints found in", args.save_dir)
                return
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
        else:
            checkpoint_path = args.checkpoint
        
        evaluate(
            checkpoint_path=checkpoint_path,
            config_path=args.config
        )
        
if __name__ == "__main__":
    main()