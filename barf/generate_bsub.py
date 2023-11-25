from model_interpolation import uniform_sampling_strategies, integration_strategies

with open("/work3/s204111/nerf-experiments/bsub_scripts/run_experiment.sh", "r") as f:
    template = f.read().split("\n")

filenames = []
for sampling_strat in uniform_sampling_strategies.__args__:
    for int_strat in integration_strategies.__args__:
        for offset in [0, -1]:
            for proposal in ["", "no-"]:
                python = f"python main_refactor.py --uniform_sampling_strategy={sampling_strat} --integration_strategy={int_strat} --uniform_sampling_offset_size={offset} --{proposal}use_proposal"
                filenames.append(f"/work3/s204111/nerf-experiments/bsub_scripts/temp_{len(filenames)}.sh")
                with open(filenames[-1], "w") as f:
                    f.write("\n".join(template + [python]))

with open("/work3/s204111/nerf-experiments/bsub_scripts/bsub_submitter.sh", "w") as f:
    f.write("\n".join([f"bsub < {filename}" for filename in filenames]))

print("bash /work3/s204111/nerf-experiments/bsub_scripts/bsub_submitter.sh")