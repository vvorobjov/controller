import datetime
import os
from timeit import default_timer as timer

import structlog
from config.MasterParams import MasterParams
from config.nrp_sim_config import SimulationConfig
from config.paths import RunPaths
from neural.plot_utils import plot_controller_outputs
from nrp_client import NrpCore
from nrp_protobuf import wrappers_pb2
from plant.plant_plotting import plot_plant_outputs
from tqdm import tqdm


def main():
    client_log = structlog.get_logger("nrp_client")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["EXEC_TIMESTAMP"] = timestamp
    print(f"env['EXEC_TIMESTAMP']={os.environ.get('EXEC_TIMESTAMP')}")

    run_paths = RunPaths.from_run_id(timestamp)
    master_config = MasterParams.from_runpaths(run_paths=run_paths)

    time_prep = master_config.simulation.time_prep
    time_move = master_config.simulation.time_move
    time_post = master_config.simulation.time_post
    n_trials = master_config.simulation.n_trials
    resolution = master_config.simulation.resolution

    SimConfig = SimulationConfig()
    SimConfig.SimulationTimeout = int(
        (time_prep + time_move + time_post) * n_trials / 1000
    )

    simconfig_path = str(run_paths.params_json).replace(".json", "_simconfig.json")
    with open(simconfig_path, "w") as f:
        f.write(SimConfig.model_dump_json(indent=2))
    client_log.debug("SimulationConfig loaded and dumped successfully.")

    nrp = NrpCore(
        "0.0.0.0:5679",
        "/sim/controller/",
        simconfig_path,
        log_output=run_paths.logs / "nrpcore_log.log",
    )

    start_time = timer()

    nrp.initialize()
    client_log.debug("Nrp server initialized successfully")

    loop_start_time = timer()
    steps_trial = int((time_prep + time_move + time_post) / resolution)
    steps_tot = steps_trial * n_trials

    client_log.info(f"Start run loop. {n_trials} trials ({steps_tot} total iterations)")

    it_step = int(steps_trial / 100)
    with tqdm(total=n_trials, desc="Total Simulation", unit="trial") as pbar_total:
        for trial_idx in range(n_trials):
            with tqdm(
                total=steps_trial,
                desc=f"Trial {trial_idx+1}",
                unit="iter",
                leave=False,
            ) as pbar_trial:

                for i in range(int(steps_trial / it_step)):
                    nrp.run_loop(it_step)
                    pbar_trial.update(it_step)

                pbar_total.update(1)

    loop_end_time = timer()
    total_loop_time = datetime.timedelta(seconds=loop_end_time - loop_start_time)
    client_log.debug(f"Simulation time: {total_loop_time.total_seconds():.1f} s")

    # shutdown nrp server
    nrp.shutdown()

    if master_config.plotting.PLOT_AFTER_SIMULATE:
        client_log.info("--- Generating Plots (Standalone) ---")
        plot_start_time = timer()
        plot_controller_outputs(run_paths)
        plot_plant_outputs(run_paths)
        plot_end_time = timer()
        total_plot_time = datetime.timedelta(seconds=plot_end_time - plot_start_time)
        client_log.info(f"Plotting Finished. {total_plot_time.total_seconds():.1f} s")

    # recap logs
    end_time = timer()
    total_time = datetime.timedelta(seconds=end_time - start_time)
    client_log.info(
        f"Simulation completed. Total execution time: {total_time.total_seconds():.1f} s"
    )


if __name__ == "__main__":
    main()
