# Running S1 Flood tutorials on NAS

**Make sure you follow the steps in [initial setup](./initial_setup.md) before running this tutorial!**

This tutorial will walk you through running the [TerraMind Sentinel-1 flood segmentation tutorial](https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb) as a job on the NAS system. The steps are as follows:

1. Enter the environment
1. Prepare input files
1. Check the queue status
1. Submit your job
1. Monitor your job
1. View stdout and stderr logs

## Entering the environment
SSH into into the Caebus front end (`ssh cfe`), navigate to your nobackup directory (`cd /nobackup/$USER`), make/enter a new terramind directory (`mkdir terramind && cd terramind`)

```bash
ssh cfe
cd /nobackup/$USER
mkdir terramind && cd terramind
```
## Prepare input files
In your `terramind` directory, clone this repository (`terramind-docs`) and copy the NAS submission script and the model Python script to the current directory:

```bash
git clone https://github.com/ASFHyP3/terramind-docs.git
cp terramind-docs/NAS/terramind_nas_submission_script.sh .
cp terramind-docs/NAS/terramind_sen1floods.py .
```

In the same directory, download the input data:
```bash
curl -L "https://drive.usercontent.google.com/download?id={1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS}&confirm=xxx" -o sen1floods11_v1.1.tar.gz
tar -xzf sen1floods11_v1.1.tar.gz
rm sen1floods11_v1.1.tar.gz
```
You should now be ready to run the model.

## Submit your job
In NAS, we run jobs using the [Portable Batch System (PBS)](https://www.nas.nasa.gov/hecc/support/kb/portable-batch-system-(pbs)-overview_126.html). We submit jobs using the `qsub` command and monitor jobs using the `qstat` command.

You will need to modify the submission script (`terramind_nas_submission_script.sh`) before running a job. Take a look at the submission script by calling `vi terramind_nas_submission_script.sh`. Lines starting with `# PBS` are configurations for PBS. Lines starting with `##` are comments that describe what each set of lines is doing.  Read through these comments. Next, move to line 36 and change the email to your email. This will allow the NAS system to notify you of your job's status. 

To submit the training job we specify the queue we want to submit to and the submission script:
```bash
qsub -q gpu_devel terramind_nas_submission_script.sh
```
This command will print your job ID to the screen. Save the first set of digits somewhere safe - you will need them to query the status of your job.

Note that the `gpu_devel` is a special queue designed for quick prototyping. Each user is only allowed to have one job in this queue at a time, and there are fewer resources available per job. **DO NOT SUBMIT PRODUCTION JOBS TO THIS QUEUE**. Use the `gpu` queue instead.

## Monitor your job
You can use the `qstat` command and your job ID (`XXXXXX`) to query the status of your job (`qstat XXXXXX`). This will produce an output like:
```
                                                       Req'd    Elap
JobID         User     Queue    Jobname        TSK Nds wallt S wallt Eff
------------- -------- -------- -------------- --- --- ----- - ----- ---
XXXXXX.pbspl4 ffwillia gpu_deve terramind-run1  16   1 01:00 R 00:00  0%
```
The most important field is the status field. The values are as follows:

| Code | State        | Meaning                                                                 |
|------|--------------|-------------------------------------------------------------------------|
| Q    | Queued       | Job is in the queue, waiting to be scheduled.                           |
| R    | Running      | Job is currently executing.                                             |
| H    | Held         | Job is held (by user or system) and will not run until released.        |
| W    | Waiting      | Job is waiting for execution time (e.g., scheduled start).              |
| T    | Transiting   | Job is being moved to/from another server.                              |
| S    | Suspended    | Job has been suspended.                                                 |
| E    | Exiting      | Job is finishing; execution is done but cleanup is in progress.         |
| F    | Finished     | Job has completed execution and left the queue (success or failure).    |

Note that `F` means finished - not failed!

If you call `qstat XXXXXX` after your job is finished, you will receive the following message:
```
qstat: XXXXXX.pbspl4.nas.nasa.gov Job has finished, use -x or -H to obtain historical job information
```
As this message states, use the `qstat -fx XXXXXX` instead to query the final results of your job. If this command reports `0` for the `Exit_status` field, congrats - your job completed successfully!

## View stdout and stderr logs
Based on lines 32 and 33 of our submission script, PBS has written the terminal output (stdout) and the error output (stderr) to the files terramind.out and terramind.err in the directory you submitted the job from. You can open these files with `vi terramind.out terramind.err` and view the results. If you had a non-zero exit code in the previous step, take a look at the terramind.err file first.
