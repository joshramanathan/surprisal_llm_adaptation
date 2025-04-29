import surprisal_llm_adaptation

def main():
    # Initialize the experiment runner with the specified model ID
    runner = surprisal_llm_adaptation.ExperimentRunner(model_id="EleutherAI/pythia-1.4b")
    
    # Run the experiment with specified parameters
    runner.run_experiment(num_proc=8, batch_size=9, block_size=128)

if __name__ == "__main__":
    main()