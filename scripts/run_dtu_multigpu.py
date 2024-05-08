# training script for DTU dataset

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

factors = [2] * len(scenes)

excluded_gpus = set([])

dataset_dir = "/mnt/nas_3dv/hdd1/datasets/DTU/DTU_mask"
output_dir = "exp_dtu/release"

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/scan{scene} -m {output_dir}/scan{scene} {common_args}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    # tsdf fusion
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1"
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s {dataset_dir}/scan{scene} -m {output_dir}/scan{scene} --iteration 30000 {common_args}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    # evaluate
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python evaluate_dtu_mesh.py -m {output_dir}/scan{scene} --iteration 30000 --scan_id {scene}"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, excludeID=[]))
        # all_available_gpus = set([0,1,2,3])
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

