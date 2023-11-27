import os
from model_interpolation import uniform_sampling_strategies, integration_strategies



###### options

template_path      = "/work3/s204111/nerf-experiments/bsub_scripts/run_experiment.sh"
tmp_scripts_path   = "/work3/s204111/nerf-experiments/bsub_scripts"



# sampling tests
main_file          = "run_sampling_test.py"
argument_iterables = [uniform_sampling_strategies.__args__, integration_strategies.__args__, [0, -1]]

def executer(sampling_strat, int_strat, offset):
    return f"python {main_file} --uniform_sampling_strategy={sampling_strat} --integration_strategy={int_strat} --uniform_sampling_offset_size={offset}"


# mip barf
main_file        = "run_mip_blur_test.py"
argument_iterables = [[]]



########### don't touch below this line

if not os.path.exists(main_file):
    raise FileNotFoundError(f"couldn't find file {main_file} in {os.getcwd()}")

def combinations_iterator(*args):
    if len(args) > 1:
        for arg0 in args[0]:
            for argrest in combinations_iterator(*args[1:]):
                yield (arg0, *argrest)
    else: yield from map(lambda x: (x,), args[0])

with open(template_path, "r") as f:
    template = [line for line in f.read().split("\n") if not "python" in line]

filenames = []
for args in combinations_iterator(*argument_iterables):
    filenames.append(os.path.join(tmp_scripts_path, f"temp_{len(filenames)}_{'_'.join(map(str, args))}.sh"))
    with open(filenames[-1], "w") as f:
        python = executer(*args)
        if not python.startswith("python"): raise ValueError("executer must return string that starts with 'python'")
        f.write("\n".join(template + [python]))

with open(os.path.join(tmp_scripts_path,"bsub_submitter.sh"), "w") as f:
    f.write("\n".join([f"bsub < \"{filename}\"" for filename in filenames]))

print(f"bash {os.path.join(tmp_scripts_path,'bsub_submitter.sh')}")