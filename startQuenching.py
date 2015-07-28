import os
import shutil
import numpy as np
import subprocess as sb

import model_builder as mdb
import project_tools as pjt

def get_slurm(jobname,nodes,ppn,time,command,queue="commons",email=False):
    slurm_string = "#!/bin/bash \n"
    slurm_string +="#SBATCH --job-name=%s\n" % jobname
    slurm_string +="#SBATCH --partition=%s\n" % queue
    slurm_string +="#SBATCH --nodes=%d\n" % nodes
    slurm_string +="#SBATCH --ntasks-per-node=%d\n" % ppn
    slurm_string +="#SBATCH --time=%s\n" % time
    slurm_string +="#SBATCH --exclusive\n"
    if email:
        slurm_string +="#SBATCH --mail-user=alexkluber@gmail.com\n"
        slurm_string +="#SBATCH --mail-type=ALL\n" 
    slurm_string +="#SBATCH --export=ALL\n\n"
    slurm_string +="cd $SLURM_SUBMIT_DIR\n"
    slurm_string +="%s\n" % command
    return slurm_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name of .ini file.")
    parser.add_argument("--nonnative_variance",
                        type=str,
                        required=True,
                        nargs="+",
                        help="Nonnative variance.")
    parser.add_argument("--replicas",
                        type=int,
                        required=,
                        nargs="+",
                        help="Nonnative variance.")
    parser.add_argument("--n_nodes",
                        type=int,
                        required=True,
                        help="Number of nodes.")
    parser.add_argument("--ppn",
                        type=int,
                        required=True,
                        help="Number of processors per node.")
    parser.add_argument("--n_jobs",
                        type=int,
                        required=True,
                        help="Number of jobs.")
    parser.add_argument("--rank_offset",
                        type=int,
                        default=0,
                        help="Increment the rank.")
    parser.add_argument("--walltime",
                        type=str,
                        default="00:10:00",
                        help="Expected walltime.")

    args = parser.parse_args()

    name = args.name
    nonnative_variance = args.nonnative_variance
    rank_offset = args.rank_offset
    n_jobs = args.n_jobs
    nodes = args.n_nodes
    ppn = args.ppn
    n_processors_per_job = nodes*ppn
    time = args.walltime

    replicas = [1]

    for n in range(len(nonnative_variance)):
        for rep in replicas:
            #shutil.copy("sim_quenching.py","random_b2_%s/replica_%d/" % (nonnative_variance[n],rep))
            os.chdir("random_b2_%s/replica_%d" % (nonnative_variance[n],rep))
            print "\nnonnative variance: ", nonnative_variance[n]

            if not os.path.exists("tables"):
                print "Saving table files"
                model,fitopts = mdb.inputs.load_model(name)
                os.mkdir("tables")
                os.chdir("tables")
                model.save_table_files()
                os.chdir("..")
            else:
                print "Don't need to save table files this time"

            for j in range(n_jobs):
                if j == 0:
                    command = "srun python -m misc.sim_quenching --name %s" % name
                else:
                    command = "srun python -m misc.sim_quenching --name %s --rank_offset %d" % (name,rank_offset)

                # Submit job script
                jobname = "%s_qnch_%d" % (name,j)
                with open("quench_parallel_%d.slurm" % j,"w") as fout:
                    # Write job script
                    slurmjob = get_slurm(jobname,nodes,ppn,time,command,queue="commons",email=True)
                    fout.write(slurmjob)

                with open("sbatch%d.out" % j,"w") as fout:
                    # Submit parallel job to queue
                    sbatch = "sbatch quench_parallel_%d.slurm" % j
                    sb.call(sbatch.split(),stdout=fout)

                rank_offset += n_processors_per_job

            os.chdir("../..")
