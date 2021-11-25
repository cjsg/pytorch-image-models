import os
import subprocess
import traceback
from datetime import datetime

def check_restart(min_hours=4, window_start=5, window_end=8):
    try:
        file_path = os.environ['_CONDOR_JOB_AD']
        jobfile = open(file_path, 'r')
        lines = jobfile.read().splitlines()

        jobads = dict()
        for line in lines:
            key, value = line.split(' = ', 1)
            jobads[key] = value

        cluster_id = jobads['ClusterId']
        proc_id = jobads['ProcId']
        bid = jobads['JobPrio']

        # "Minimum running time"-condition
        condor_q_output = subprocess.check_output(f'condor_q {cluster_id}.{proc_id} -run -currentrun', shell=True, text=True)
        run_time = condor_q_output.splitlines()[-1].split(' ')[-2]  # RUN_TIME -> f.ex. 0+01:35:15
        days, time = run_time.split('+', 1)
        hours, minutes, sec = time.split(':')
        total_hours = int(days) * 24 + int(hours) + int(minutes) / 60 + int(sec) / 3600
        min_time_cond = (total_hours >= min_hours)

        # "Time window with low prob of job submission"-condition
        now = datetime.now()
        window_start = now.replace(hour=window_start, minute=0, second=0, microsecond=0)
        window_end = now.replace(hour=window_end, minute=0, second=0, microsecond=0)
        time_window_cond = (window_start <= now <= window_end)

        # "No competing job"-condition
        idle_jobs = subprocess.check_output(
            f"condor_q -run -constraint 'JobPrio >= {bid} && JobStatus =?= 1'", shell=True, text=True)  #  "&& requestgpus >= 1 && Owner =!= whoami'"
        idle_jobs_list = idle_jobs.splitlines()
        no_competition_cond = (len(idle_jobs_list) == 4) and (idle_jobs_list[-1] == ' ID      OWNER            SUBMITTED     RUN_TIME HOST(S)')

        return min_time_cond and time_window_cond and no_competition_cond

    except Exception as e:
        traceback.print_exc()
        print('Not restarting')
        return False

if __name__ == '__main__':
    print(check_restart())
    print(check_restart(2, 17, 18))
