#!/usr/bin/env python3
import datetime
import os
from pathlib import Path
from timeit import default_timer as timer

from neural.nest_adapter import initialize_nest, nest

initialize_nest("MUSIC")


import structlog
from config.paths import RunPaths
from mpi4py import MPI
from mpi4py.MPI import Comm
from neural.Controller import Controller
from neural.data_handling import collapse_files, save_conn_weights
from neural.plot_utils import plot_controller_outputs
from utils_common.log import setup_logging

from complete_control.config.core_models import SimulationParams
from complete_control.config.MasterParams import MasterParams
from complete_control.neural.Controller import Controller
from complete_control.neural.data_handling import collapse_files
from complete_control.neural_simulation_lib import (
    create_controllers,
    setup_environment,
    setup_nest_kernel,
)


def run_simulation(
    master_config: MasterParams,
    path_data: Path,
    controllers: list[Controller],
    comm: MPI.Comm,
):
    """Runs the NEST simulation for the specified number of trials."""

    log: structlog.stdlib.BoundLogger = structlog.get_logger(
        "main.simulation_loop", log_all_ranks=True
    )
    single_trial_ms = master_config.simulation.duration_single_trial_ms
    n_trials = master_config.simulation.n_trials

    # --- Prepare for Data Collapsing ---
    pop_views = []
    for controller in controllers:
        pop_views.extend(controller.get_all_recorded_views())

    log.info("collected all popviews")
    controller = controllers[0]
    log.info("Starting Simulation")
    ###############################
    log.info("Waiting for all MPI ranks to finish MUSIC port setup before Prepare")
    MPI.COMM_WORLD.barrier()
    ###############################
    nest.Prepare()
    for trial in range(n_trials):
        current_sim_start_time = nest.GetKernelStatus("biological_time")
        log.info(
            f"Starting Trial {trial + 1}/{n_trials}",
            duration_ms=single_trial_ms,
            current_sim_time_ms=current_sim_start_time,
        )
        log.info(f"Current simulation time: {current_sim_start_time} ms")
        start_trial_time = timer()

        nest.Run(single_trial_ms)

        if controller.use_cerebellum:
            controller.record_synaptic_weights(trial)

        end_trial_time = timer()
        trial_wall_time = datetime.timedelta(seconds=end_trial_time - start_trial_time)
        log.info(
            f"Finished Trial {trial + 1}/{n_trials}",
            sim_time_end_ms=nest.GetKernelStatus("biological_time"),
            wall_time=str(trial_wall_time),
        )
    nest.Cleanup()
    log.info("--- All Trials Finished ---")
    nest.SyncProcesses()

    # --- Data Collapsing (after all trials) ---
    log.info("Attempting data collapsing for all trials...")
    start_collapse_time = timer()
    collapse_files(
        path_data,
        pop_views,
        comm,
    )
    if controller.use_cerebellum and master_config.SAVE_WEIGHTS_CEREB:
        log.info("Saving recorded synapse weights for all trials started...")
        save_conn_weights(
            controller.weights_history,
            path_data,
            "weightrecord",
        )

    end_collapse_time = timer()
    collapse_wall_time = datetime.timedelta(
        seconds=end_collapse_time - start_collapse_time
    )
    log.info(
        "Data collapsing for all trials finished",
        wall_time=str(collapse_wall_time),
    )

    log.info("--- Simulation Finished ---")


def coordinate_paths_with_receiver() -> tuple[str, RunPaths]:
    shared_data = {
        "timestamp": None,
        "paths": None,
    }
    run_timestamp_str = None
    if rank == 0:
        shared_data["timestamp"] = run_timestamp_str = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        shared_data["paths"] = RunPaths.from_run_id(run_timestamp_str)
        print("sending paths to all processes...")

    shared_data = MPI.COMM_WORLD.bcast(shared_data, root=0)
    run_timestamp_str = shared_data["timestamp"]
    run_paths: RunPaths = shared_data["paths"]

    return run_timestamp_str, run_paths


def generate_param_plot(data, pop_name, param_name, plots_path, plot_one_n=True):
    import numpy as np
    import matplotlib.pyplot as plt

    senders = np.array(data["senders"])
    times = np.array(data["times"])
    param = np.array(data[param_name])

    fig = plt.figure(figsize=(10, 6))

    # take one neuron (id=sender)
    if plot_one_n:
        gid = np.unique(senders)[0]
        mask = senders == gid
        plt.plot(times[mask], param[mask])
        plt.title(f"{pop_name} - {param_name} - neuron {gid}")
        # if param_name == "CV_fbk" or param_name == "CV_pred":
        #    plt.ylim(0, 1)

    # plot all neurons
    else:
        for gid in np.unique(senders):
            mask = senders == gid
            plt.plot(
                times[mask], param[mask], color=(0, 0, 0, 0.05), label=f"Neuron {gid}"
            )
        plt.title(f"{pop_name} - {param_name} - all neurons")
        # plt.legend()

    plt.xlabel("Time (ms)")
    plt.ylabel(param_name)

    fig.savefig(plots_path / f"{pop_name}_{param_name}.png")
    plt.close(fig)
    return


if __name__ == "__main__":

    comm = MPI.COMM_WORLD.Create_group(
        MPI.COMM_WORLD.group.Excl([MPI.COMM_WORLD.Get_size() - 1])
    )
    rank = comm.rank
    run_timestamp_str, run_paths = coordinate_paths_with_receiver()
    run_paths = RunPaths.from_run_id(run_timestamp_str)

    setup_logging(
        MPI.COMM_WORLD,
        log_dir_path=run_paths.logs,
        timestamp_str=run_timestamp_str,
        log_level=os.environ.get("LOG_LEVEL", "DEBUG"),
    )

    main_log: structlog.stdlib.BoundLogger = structlog.get_logger("main")
    main_log.info(
        f"Starting Standalone Run: {run_timestamp_str}",
        run_dir=str(run_paths.run),
        log_all_ranks=True,
    )
    main_log.info(
        "MPI Setup Complete (Standalone)",
        world_rank=MPI.COMM_WORLD.Get_rank(),
        world_size=MPI.COMM_WORLD.Get_size(),
        sim_rank=comm.rank,
        sim_size=comm.size,
        log_all_ranks=True,
    )

    start_script_time = timer()

    # Load master config
    master_config = MasterParams.from_runpaths(run_paths=run_paths, USE_MUSIC=True)
    with open(run_paths.params_json, "w") as f:
        f.write(master_config.model_dump_json(indent=2))
    main_log.info("MasterParams initialized in music_start_sim (MUSIC).")

    # Setup environment and NEST kernel
    setup_environment(master_config)
    setup_nest_kernel(
        master_config,
        run_paths.data_nest,
    )

    # Create controllers
    controllers = create_controllers(master_config, comm=comm)

    #######################################################
    # create and connect mm to state
    params_state = [
        "var_fbk",
        "var_pred",
        "mean_fbk",
        "mean_pred",
        "w_fbk",
        "w_pred",
        "CV_fbk",
        "CV_pred",
    ]
    mm_state = nest.Create("multimeter", {"record_from": params_state})
    nest.Connect(mm_state, controllers[0].pops.state_p.pop)

    mm_pred = nest.Create("multimeter", 1, {"record_from": ["lambda_poisson"]})
    mm_fbk = nest.Create("multimeter", 1, {"record_from": ["lambda_poisson"]})
    for x in range(200):
        if (
            "lambda_poisson"
            in nest.GetStatus(controllers[0].pops.pred_p.pop[x])[0].keys()
        ):
            nest.Connect(mm_pred, controllers[0].pops.pred_p.pop[x])
        else:
            print(f"Neuron pred {x} has not been connected")
    for x in range(200):
        if (
            "lambda_poisson"
            in nest.GetStatus(controllers[0].pops.fbk_smooth_p.pop[x])[0].keys()
        ):
            nest.Connect(mm_fbk, controllers[0].pops.fbk_smooth_p.pop[x])
        else:
            print(f"Neuron fbk_smooth {x} has not been connected")

    print(
        f"Dcn params: {nest.GetStatus(controllers[0].cerebellum_handler.cerebellum.populations.forw_dcnp_p_view.pop[0])[0].keys()}"
    )
    ##############################################################

    # Run simulation
    run_simulation(master_config, run_paths.data_nest, controllers, comm)

    ##########################à

    main_log.info("Start - Params from mm printed successfully")
    pop_name = "State_Estimator"
    plots_path = run_paths.figures
    data_state = nest.GetStatus(mm_state, "events")[0]
    data_pred = nest.GetStatus(mm_pred, "events")[0]
    data_fbk = nest.GetStatus(mm_fbk, "events")[0]
    for param in params_state:
        try:
            generate_param_plot(
                data_state, pop_name, param, plots_path, plot_one_n=True
            )
        except Exception as e:
            print(f"Error: {e}, \n param_: {param}")
    try:
        pop_name = "Pred"
        param = "lambda_poisson"
        generate_param_plot(data_pred, pop_name, param, plots_path, plot_one_n=False)
        pop_name = "FbkSmooth"
        generate_param_plot(data_fbk, pop_name, param, plots_path, plot_one_n=False)
    except Exception as e:
        print(f"Pred Error: {e} \n")
    main_log.info("Params from mm printed successfully")

    ###############################################################

    # Plotting (Rank 0 Only)
    if rank == 0 and master_config.PLOT_AFTER_SIMULATE:
        main_log.info("--- Generating Plots (Standalone) ---")
        start_plot_time = timer()
        plot_controller_outputs(run_paths)
        end_plot_time = timer()
        plot_wall_time = datetime.timedelta(seconds=end_plot_time - start_plot_time)
        main_log.info(f"Plotting Finished (Standalone)", wall_time=str(plot_wall_time))

    # Final Timing
    end_script_time = timer()
    main_log.info(f"--- Script Finished (Standalone) ---")
    main_log.info(
        f"Total wall clock time: {datetime.timedelta(seconds=end_script_time - start_script_time)}"
    )
