# Making the most of your jobs in CC

So! Now that you have everything set up, you can begin running small experiments.
I won't go into all the details of how to write CC scripts. If you want a
more basic walkthrough of that, I highly recommend briefly reading through
[this page](https://docs.computecanada.ca/wiki/Running_jobs) of the CC docs.

The general workflow for how you should be running experiments with CC is:
1. Test and make sure everything's working locally.
2. Gauge how much compute you'll need for each experiment.
3. Write your script for running all experiments.
4. Gauge how much time you'll need for each set of experiments.
5. Run your experiments!

The example I'll give here is for research that I'm involved in - reinforcement
learning - but can be broadly applied to any experiments that need compute.

### 1. Test and get working locally
With that, let's start with `1.`. Simply having your script run on your machine does not mean
that it's working. I'll give a few examples for what I mean by working and 
how you can iteratively build up to that.

In supervised learning, you normally perform updates based on batches of
samples of the data. The first thing you should always do with a complicated
model is to make sure you can overfit on a SINGLE example. Once that's done
try to fit the data on a batch. If that's fine, then move onto your full dataset.

In reinforcement learning, what you could do instead is to construct a toy
problem that you know the optimal policy/optimal value function to. It could
be something as simple as an MDP with a single action and only a few states
(in the case of learning a value function). After that's working you probably
want to try one of the classic control environments (if you're doing control) 
like cartpole or mountain car. Scaling up in the reinforcement learning case
tends to be quite a bit more difficult, because "scaling" isn't clearly defined
in this case. Is Atari a scaled up environment? Or is it also lacking/too much?

To sum up both examples, start small first. From there, slowly build up until
you reach a point that you think will be good enough to present as results
for a solid experiment. A lot of the time you'll be able to use your smaller
experiments in your work as well, so it acts doubly as a sanity check and also
potential results for you. In this case, "working" means working at the scale that you want to experiment
in.

At this point, if you do research in ML/RL, you'll also want to start thinking
about hyperparameters you want to sweep over, and try a few runs locally with 
a few of those settings. Try and get a diverse, representative group of experiments
to test for divergence/NaNs. Working also means NaN's don't show up in your learned parameters.

One last piece of advice - while writing the code for experiments, try to avoid
dynamic memory/VRAM allocation as much as possible, and try to use as little
VRAM as possible. An example of this would be 
implementing a [replay buffer](https://paperswithcode.com/method/experience-replay).
What I've seen some people do is to simply have an empty list where they append
(action, reward, next_state, gamma) pairs. The issue with this is it's hard
to gauge how much memory you actually need at the end of your script. A better
idea would be to save state, action, reward and gamma into separate contiguous
numpy/pytorch/tensorflow arrays, that are instantiated with zeros (make sure
you type correctly when saving! For example when saving images, you
can save each number as a uint8 integer). After instantiation,
when you receive a tuple of experience, you can simply bring everything off your
GPU and index into these arrays to save your experience. This way your buffer
doesn't grow as your experiment runs.

### 2. Gauging Compute
So now that everything's nice and working locally, you need to start thinking
about the resources you'll need to run your experiments.

It's super easy to get caught up in your algorithm development, but this is also
an important part of being a good researcher. In CC, each script has you 
specify how much memory, CPUs, GPUs and time each script needs to run. You need
to find these out from the scripts you wrote.

#### 2.1 GPU memory requirements
What I like to do is to start with GPU requirements and work from there - since
GPU resources are normally the hardest things to come by. It turns out you can
perform multiple experiments on the same GPU without much/any loss in performance.
You just need to make sure that all those experiments don't exceed the amount of
VRAM on the GPU.

There are a few ways of gauging GPU usage during a run. If you're coding in Python, I 
highly recommend the pip package [pyNVML](https://pypi.org/project/nvidia-ml-py3/).
This allows you to more accurately track peak memory usage of some script you're running
if you ping the GPU stats correctly. Here's an example of how I do it in code:
```python
import nvidia_smi
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
max_gpu_mem = 0
...
# Within your training loop
res = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
max_gpu_mem = max(max_gpu_mem, res.used)
```
In this snippet, `max_gpu_mem` should indicate (in bytes) the peak memory usage
of the script you're running. 

A non-programmatic way of doing this is simply running your script in one 
[tmux](https://github.com/tmux/tmux/wiki) session and running `nvidia-smi`
in another pane. The only issue here is these GPU memory usage numbers may be
slightly off - I'm not entirely sure why, but it's an issue I've run into.

Another note to remember - different algorithms have different VRAM requirements.
Remember to factor this into account when you calculate how many/which jobs 
can be mapped to which GPU.

With the max memory usage for one experiment figured out, it's simply a matter
of finding out how much VRAM is available to you when running experiments
to figure out how many experiments you can run on a single GPU. If
you're on [Compute Canada](README.md), the GPU sizes are listed
[here](https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm), so make sure
to pick a GPU type and stick with it when you allocate GPUs in your script.

#### 2.2 Other requirements
All the other requirements for your job should fall into place quite nicely after
figuring out how many experiments you can run per GPU. In terms of memory requirements,
I normally just run an experiment with [htop](https://htop.dev/) open and track
how much memory is being used for each experiment. As for CPU requirements, you could simply stick
to 1 CPU per experiment if there isn't much parallelization of your python scripts.
I personally wouldn't go any less than 1 CPU per experiment - it makes timing your
experiments that much harder.

### 3. Writing your experiment scripts

This is the part that I've seen the most variation with between researchers
that use Compute Canada. With that in mind, throughout this section I'll be
talking about what I do in order to generate these scripts and give brief
descriptions of the tools I'll be using to do this - use this as a general
guideline and tweak it to however you think will fit your needs.

#### 3.1 Figuring out which hyperparameters to combine
First off, you need to make sure you're able to list experiment arguments out from the
command line when you run your python script. For example, if your experiment
run script is `run.py`, you need to be able to add arguments to this, such as
`python run.py --step-size 0.001`. I would take a look at [argparse](https://docs.python.org/3/howto/argparse.html)
for how to do this in Python.

Next, what I like to do is to create a text file (per CC job) that lists (one per line)
the different run scripts I'd like to run. I call these task files.
Here's an example of the contents of a task file `tasks.txt`:
```bash
python run.py --step-size 0.01 --seed 1
python run.py --step-size 0.01 --seed 2
python run.py --step-size 0.01 --seed 3
python run.py --step-size 0.001 --seed 1
...
```
(Of course you should always run more than 3 seeds in an experiment!)

To do this, I like to write another python script (which I'll call `write_jobs.py`)
that when run will generate this task file for me. I highly recommend taking a look
at the [product function in itertools](https://docs.python.org/3/library/itertools.html#itertools.product) 
built in to Python.

#### 3.2 Multiple task files
Now you have to figure out how many experiments should I put in one task file.
You could simply put all your experiments in one task file and run that on CC,
but the issue is fitting all of your experiments onto some number of GPUs.
If all your experiments don't fit on a single GPU, then in this case what you'll need
to do is to request the requisite number of GPUs for all your experiments
and map each experiment to a corresponding GPU. While this is fine, what I prefer
to do is to divvy up experiments into CC jobs with 1 GPU per job. Let me give
a slightly more concrete example:

Let's say your experiment peaks at VRAM usage of less than 2 GB (always round up), and after
considering all your hyperparameters, you have 100 experiments to run. From
the [CC webpage](https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm)
you decide to use the `p100` GPUs for your experiments. Each one of these
GPUs have 12 GB of VRAM. That means you can fit 12 / 3 = 4 experiments onto a single
GPU at one time. This means that I would split up my task files into files with 
4 experiments each (with 100 / 4 = 25 task files total), 
and run one CC job for each of these task files. You could quite easily achieve
this in Python by keeping a count of the number of experiments written and open a
new task file whenever that count modulo the number of experiments per job is 0.

#### 3.3 GNU Parallel
Finally, we reach the point where we can write our actual [CC run script](https://docs.computecanada.ca/wiki/Running_jobs).
Since for each job, you want to run all your experiments in parallel, we need to
utilize [GNU parallel](https://docs.computecanada.ca/wiki/GNU_Parallel). This is a way
of starting multiple processes across multiple CPUs. To achieve this, since all the
experiments we want to run in parallel are nicely written line-by-line in these task files,
all you simply need to do is call
```bash
parallel < /path/to/tasks.txt
```
to run all experiments in `tasks.txt` in parallel. You can also add the `--joblog` option
to specify where to log the run results of each experiment - this will also tell you which 
experiments failed to run by the end of your CC job. Remember to make sure you activate
your [virtual env](https://docs.python.org/3/tutorial/venv.html) before this parallel call.

So now our CC script is just missing one thing - the amount of time to run each job.

### 4. How much time do we need for our experiments?
This used to be the biggest pain point for me when I first started running experiments
on CC. It turns out it's somewhat easier than you would expect. What you need is an
estimate of how much time your job will take to run. Assuming all these jobs run in
parallel, we can extrapolate this from how long it takes to run a single experiment.

For most experiments, it's always a good idea to do some logging at some given increment.
For supervised learning this makes the most sense between every epoch, or even after every
given period of updates. Similarly for reinforcement learning, you might log experiment
statistics after every episode or after every given period of updates. We're going to
stick to a given period of updates, because then we log after every set number of steps, 
rather than rely on episode length which can be a random variable.

In both cases, it's a good idea to set a maximum number of updates/environment steps.
With these two pieces of information, you can calculate the number of logs remaining in
your experiment (remaining steps / log period) and multiply that by a moving average
of how long it takes to go through one log period of steps/updates. This gives you
a rough estimate of how much time is left in the experiment.

To get an even better estimate of how long this will take on the CC machines, I like to
`salloc` (check out [interactive jobs](https://docs.computecanada.ca/wiki/Running_jobs#Interactive_jobs)) 
a single CPU, the GPU I'll be using and the amount of memory needed for a single
experiment. After I'm allocated this compute, I'll try running a single experiment
in this interactive job and see how much time it takes to run the experiment based on the 
above estimates.

### 5. Running your experiments
So now you'll have a CC script that looks something like this (let's call it `run_gpu.sh`):
```bash
#!/bin/sh

#SBATCH --account=TO_FILL_ACCOUNT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=someone@domain.com
#SBATCH --error=/home/TO_FILL_USER/scratch/log/slurm-%j-%n-%a.err
#SBATCH --output=/home/TO_FILL_USER/scratch/log/slurm-%j-%n-%a.out
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-24:00

cd ../../  # Go to main project folder
source venv/bin/activate  # activate venv
parallel --jobs 4 --joblog ../log/'task_'"$SLURM_ARRAY_TASK_ID"'.log' -u < 'script/interference/tasks_'"$SLURM_ARRAY_TASK_ID"'.txt'
```
Note that we use the `-u` option, which stands for ungrouped. This means that the output
of all the tasks won't be grouped into one output stream, and that you'll get logs in your
CC logs to tell you if something went wrong.
Now I'll have to briefly explain what `$SLURM_ARRAY_TASK_ID` is. This is a script to run
a [job array](https://docs.computecanada.ca/wiki/Job_arrays) - or multiple CC jobs that
indexes into multiple files. My `write_job.py` script has it so that I have multiple task files.
I'll run this job array with the following line (for 20 `tasks_{id}.txt` files):
```bash
sbatch --array=0-20 run_gpu.sh
```

But, before I do this, I like to do a sanity check to see that all my resource estimations
aren't too far off. I'll `salloc` another interactive job with the same resources as listed
in my CC script (like above) and try running the parallel call (after activating virtual
env) in the CC script. I'll let it run for at least until I see my GPU and Memory usage
level out in my experiment (with a combination of `tmux`, `htop` and `nvidia-smi`) just
to be sure I don't run into any out of memory errors.

One final note - while it is possible to run multiple experiments on a single GPU, it turns out there's
a decent amount of overhead when you start to add too many experiments on one GPU.
Try to find GPUs with less VRAM but still decently fast and run less experiments per GPU.

Finally, when this is all good to go, I'll run my script.

