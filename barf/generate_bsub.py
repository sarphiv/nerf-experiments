import os
from model_interpolation import uniform_sampling_strategies, integration_strategies



################################### options

template_path      = "/work3/s204111/nerf-experiments/bsub_scripts/run_experiment.sh"
tmp_scripts_path   = "/work3/s204111/nerf-experiments/bsub_scripts"



###### sampling tests
main_file          = "run_sampling_test.py"
# argument_iterables = [[23423,3446,444444], uniform_sampling_strategies.__args__[::-1], integration_strategies.__args__, [0, -1]]
argument_iterables = [[23423,3446,444444], uniform_sampling_strategies.__args__[::-1], ["middle"], [0, -1]]

# def executer_helper(sampling_strat, int_strat, offset, seed):
#     return True, f"python {main_file} --uniform_sampling_strategy={sampling_strat} --integration_strategy={int_strat} --uniform_sampling_offset_size={offset} --seed={seed}"

# def executer_all(seed, sampling_strat, int_strat, offset):

###################### # mip barf


# main_file        = "run_mip_barf_test.py"
# start_pixel_width_sigmas = [200, None]
# start_blur_sigmas = [0, 3, 15]
# camera_noise_sigmas = [0.15, 0.3]
# n_blur_sigmass = [2, 10]
# seeds = [12312]#,422,1114]#,2]


# argument_iterables = [seeds, start_pixel_width_sigmas, start_blur_sigmas, n_blur_sigmass, camera_noise_sigmas]

# def executer(seed, start_pixel_width_sigma, start_blur_sigma, n_blur_sigmas, camera_noise_sigma):

#     if start_pixel_width_sigma is None:
#         start_pixel_width_sigma = start_blur_sigma
#         if n_blur_sigmas == 2 or start_blur_sigma == 0:
#             return False, "Nej"

#     if n_blur_sigmas == 2:
#         if start_blur_sigma != max(start_blur_sigmas):
#             return False, "Niks"
#         else:
#             max_blur_sigma = start_blur_sigma * 2
#     else:
#         max_blur_sigma = start_blur_sigma

#     return True, f"python {main_file} --camera_origin_noise_sigma={camera_noise_sigma} --camera_rotation_noise_sigma={camera_noise_sigma} --start_blur_sigma={start_blur_sigma} --seed={seed} --start_pixel_width_sigma={start_pixel_width_sigma} --n_blur_sigmas={n_blur_sigmas} --max_blur_sigma={max_blur_sigma}"


########################### don't touch below this line

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
    execute, python = executer(*args)
    if not execute: continue
    filenames.append(os.path.join(tmp_scripts_path, f"temp_{len(filenames)}_{'_'.join(map(str, args))}.sh"))
    with open(filenames[-1], "w") as f:
        if not python.startswith("python"): raise ValueError("executer must return string that starts with 'python'")
        f.write("\n".join(template + [python]))

with open(os.path.join(tmp_scripts_path,"bsub_submitter.sh"), "w") as f:
    f.write("\n".join([f"bsub < \"{filename}\"" for filename in filenames][::-1]))

print(f"bash {os.path.join(tmp_scripts_path,'bsub_submitter.sh')}")