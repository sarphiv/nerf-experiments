import os
from model_interpolation import uniform_sampling_strategies, integration_strategies

def executer(*args):
    print("WARN: called not working executer")
    return False, "nothin"

# helper function for combining argument iterables.
def combinations_iterator(*args):
    if len(args) > 1:
        for arg0 in args[0]:
            for argrest in combinations_iterator(*args[1:]):
                yield (arg0, *argrest)
    else: yield from map(lambda x: (x,), args[0])

# DOCUMENTATION
"""
This is a easy way of basing multiple bsub scripts on one template

It generates multiple bsub scripts, and then it creates the file bsub_submitter.sh
which looks like the following:
    bsub < experiment_0
    bsub < experiment_1
    bsub < experiment_2
    .
    .
    .

then to start the experiments, run 
    bash bsub_submitter

Details:
---------

For this script to work, you must specify two variables: args_iter and executer.

 * args_iter:
    this is a list of either strings or iterables - one element in the list
    for each experiment.

    if it is a list of strings, it uses these directly to execute main file,
    i.e. the line in bsub that starts with python.

    if it is a list of iterables, these are unpacked and passed to the executer (see below)

 * executer(*args):
    This is a function that takes in the args (each element in args_iter unpacked)
    and produces the line to execute (and whether or not to execute that line)
    if you wish to omit an experiment, make an if statement in the executer, 
    such that it returns False,None for all the experiemnts that should not be run.


example:
    say we have two parameters we wish to investigate:
     - gaussian_sigma which can be [0,10]
     - camera_noise which can be [0,0.15]

    to run 4 experiments with all combinations of these, there are two options:

    option 1:
    specify args_iter manually with strings (in which case executer is ignored):

        args_iter = [
            f"python {main_file} --gaussian_sigma=0 --camera_noise=0"
            f"python {main_file} --gaussian_sigma=0 --camera_noise=0.15"
            f"python {main_file} --gaussian_sigma=10 --camera_noise=0"
            f"python {main_file} --gaussian_sigma=10 --camera_noise=0.15"

    option 2:
    specify args_iter with arguments and specify executer

        args_iter = [( 0, 0.15),
                    ( 0, 0.0),
                    (10, 0.15),
                    (10, 0.0)]

        def executer(gaussian_sigma, camera_noise):
            return True, f"python {main_file} --gaussian_sigma={gaussian_sigma} --camera_noise={camera_noise}"

        NOTE: in the case the option (0,0) is not wished to be run specify executer as follows:
        def executer(gaussian_sigma, camera_noise):
            if gaussian_sigma==0 and camera_noise==0:
                return False, None
            return True, f"python {main_file} --gaussian_sigma={gaussian_sigma} --camera_noise={camera_noise}"



The function combinations_iterator(*argument_iterables) is a helper function
    it takes in a list/tuple of iterables, and then it generates
    all the possible combinations. That is, and easy way to generate the args_iter from 
    option 2 in the example above is

    args_iter = combinations_iterator([0,10], [0, 0.15])

    This is especially helpful if there are many combinations

"""

################################### options

template_path      = "/work3/s204111/nerf-experiments/bsub_scripts/run_experiment.sh" # template bsub script to base the scripts on
tmp_scripts_path   = "/work3/s204111/nerf-experiments/bsub_scripts" # where to write the temporary bsub scripts generated by this script

################################### end of options


##################### Experiments:
# comment in/out the experiments you which to run.



#################### sampling tests


# main_file          = "run_sampling_test.py"
# # argument_iterables = [[23423,3446,444444], uniform_sampling_strategies.__args__[::-1], integration_strategies.__args__, [0, -1]]
# args_iter = combinations_iterator([23423,3446,444444], uniform_sampling_strategies.__args__[::-1], ["middle"], [0, -1])

# args_iter = [
#     f"python {main_file} --uniform_sampling_strategy=equidistant --integration_strategy=middle --uniform_sampling_offset_size=-1 --seed=5436456",
#     f"python {main_file} --uniform_sampling_strategy=equidistant --integration_strategy=middle --uniform_sampling_offset_size=-1 --seed=23423",
#     f"python {main_file} --uniform_sampling_strategy=equidistant --integration_strategy=middle --uniform_sampling_offset_size=-1 --seed=7886",
# ]



# def executer(seed, sampling_strat, int_strat, offset):
#     return True, f"python {main_file} --uniform_sampling_strategy={sampling_strat} --integration_strategy={int_strat} --uniform_sampling_offset_size={offset} --seed={seed}"

###################### # mip barf


main_file        = "run_bip_barf.py"
start_pixel_width_sigmas = [None]
start_blur_sigmas = [0, 20, 100, 200]
camera_noise_sigmas = [0.15]
n_blur_sigmass = [10]
seeds = [12312]#,422,1114]#,2]


argument_iterables = [seeds, start_pixel_width_sigmas, start_blur_sigmas, n_blur_sigmass, camera_noise_sigmas]

def executer(seed, start_pixel_width_sigma, start_blur_sigma, n_blur_sigmas, camera_noise_sigma, ):
    if start_pixel_width_sigma is None: start_pixel_width_sigma = start_blur_sigma
    return True, f"python {main_file} --seed {seed}  --start_pixel_width_sigma {start_pixel_width_sigma}  --start_blur_sigma {start_blur_sigma}  --n_blur_sigmas {n_blur_sigmas}  --camera_origin_noise_sigma {camera_noise_sigma} --camera_rotation_noise_sigma {camera_noise_sigma}"  

args_iter = combinations_iterator(*argument_iterables)

########################### don't touch below this line ############################################

if not os.path.exists(main_file):
    raise FileNotFoundError(f"couldn't find file {main_file} in {os.getcwd()}")

with open(template_path, "r") as f:
    template = [line for line in f.read().split("\n") if not "python" in line]

filenames = []
for args in args_iter:
    if isinstance(args, str): execute, python = True, args
    else:                     execute, python = executer(*args)
    if not execute: continue
    filenames.append( os.path.join(tmp_scripts_path, f"temp_{len(filenames)}_{python}.sh"))
    with open(filenames[-1], "w") as f:
        if not python.startswith("python"): raise ValueError("executer must return string that starts with 'python'")
        f.write("\n".join(template + [python]))

with open(os.path.join(tmp_scripts_path,"bsub_submitter.sh"), "w") as f:
    f.write("\n".join([f"bsub < \"{filename}\"" for filename in filenames]))

print(f"bash {os.path.join(tmp_scripts_path,'bsub_submitter.sh')}")