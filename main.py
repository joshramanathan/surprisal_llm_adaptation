import surprisal_llm_adaptation

def main():
    # Initialize the experiment runner with the specified model ID
    runner = surprisal_llm_adaptation.ExperimentRunner(model_id="EleutherAI/pythia-160m-deduped")
    
    # Run the experiment with specified parameters
    runner.run_experiment(num_proc=24,
                          batch_size=100,
                          per_device_train_batch_size=100,
                          block_size=2048)

if __name__ == "__main__":
    main()