if __name__ == "__main__":
    
    import argparse
    import yaml
    from experiment import Trainer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()