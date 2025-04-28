import surprisal_llm_adaptation

runner = surprisal_llm_adaptation.ExperimentRunner(model_id="EleutherAI/pythia-1.4b")
runner.run_experiment(num_proc=8, batch_size=9, block_size=128)