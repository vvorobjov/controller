"""Motor cortex class"""

from abc import ABC, abstractmethod

from config.module_params import M1MockConfig, MotorCortexModuleConfig

# from M1MotorCortexEprop import M1MotorCortexEprop
from neural.nest_adapter import nest

from .population_view import PopView


class M1SubModule(ABC):
    @abstractmethod
    def connect(self, source_population):
        """Connect source to this component"""
        pass

    @abstractmethod
    def get_output_pops(self):
        """Return output populations (pos/neg)"""
        pass


class M1Mock(M1SubModule):
    #                motorcommands.txt
    #                        │
    #  ┌─────────┐    ┌──────┼──────────────────────────┐
    #  │         │    │      ▼         Motor Cortex     │
    #  │ Planner │    │  ┌─────────┐       ┌─────────┐  │
    #  │         │----│-▶│  Ffwd   │──────▶│   Out   │  │
    #  │         │    │  │(tracking│       │  (basic │  │
    #  └─────────┘    │  │ neuron) │       │  neuron)│  │
    #       │         │  └─────────┘       └─────────┘  │
    #       │         │                         ▲       │
    #       │  +      │  ┌─────────┐            │       │
    #       └────▶█───┼─▶│   Fbk   │            │       │
    #             ▲   │  │  (diff  │────────────┘       │
    #           - │   │  │  neuron)│                    │
    #             │   │  └─────────┘                    │
    #                 └─────────────────────────────────┘
    def __init__(self, numNeurons, motorCommands, params: M1MockConfig):
        self.N = numNeurons
        self.params = params
        self.motorCommands = motorCommands
        self.create_network()

    def create_network(self):
        par_m1 = {"base_rate": self.params.m1_base_rate, "kp": self.params.m1_kp}
        p = nest.Create("tracking_neuron_nestml", n=self.N, params=par_m1)
        nest.SetStatus(
            p,
            {
                "pos": True,
                "traj": self.motorCommands,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.output_p = PopView(p, to_file=True, label="mc_m1_p")

        n = nest.Create("tracking_neuron_nestml", n=self.N, params=par_m1)
        nest.SetStatus(
            n,
            {
                "pos": False,
                "traj": self.motorCommands,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.output_n = PopView(n, to_file=True, label="mc_m1_n")

    def connect(self, source):
        return

    def get_output_pops(self):
        return self.output_p, self.output_n


class MotorCortex:
    #                 ┌─────────────────────────────────┐
    #  ┌─────────┐    │                Motor Cortex     │
    #  │ Planner │    │  ┌─────────┐       ┌─────────┐  │
    #  └─────────┘----│-▶│    M1   │──────▶│   Out   │  │
    #       │         │  └─────────┘       └─────────┘  │
    #       │         │                         ▲       │
    #       │  +      │  ┌─────────┐            │       │
    #       └────▶█───┼─▶│   Fbk   │            │       │
    #             ▲   │  │  (diff  │────────────┘       │
    #           - │   │  │  neuron)│                    │
    #             │   │  └─────────┘                    │
    #                 └─────────────────────────────────┘
    """Module that creates the motor commands

    The MotorCortex has 2 main roles:
    1. translating the Planner's desired trajectory (rate-based) into motor commands
    2. correct motor commands accounting for current state (compared to expected state
        acc to Planner trajectory)

    1 is accomplished by the M1 Submodule. The M1 exists in two versions. A complete one
    based on E-Prop, implemented in https://github.com/shimoura/motor-controller-model;
    and a mock one, implemented above.
    """

    def __init__(self, numNeurons, mtCmds, params: MotorCortexModuleConfig):
        self.motorCommands = mtCmds
        self.N = numNeurons
        self.params = params
        self.create_net(params, numNeurons)

    def create_net(self, params: MotorCortexModuleConfig, numNeurons):
        if params.use_m1_eprop:
            pass
            # self.m1 = M1MotorCortexEprop()
        else:
            self.m1 = M1Mock(numNeurons, self.motorCommands, params.m1_mock_config)

        par_fbk = {"base_rate": params.fbk_base_rate, "kp": params.fbk_kp}
        par_out = {"base_rate": params.out_base_rate, "kp": params.out_kp}
        buf_sz = params.buf_sz

        self.m1_out_p, self.m1_out_n = self.m1.get_output_pops()
        self.fbk_p = None
        self.fbk_n = None
        self.out_p = None
        self.out_n = None

        ############ FEEDBACK ############
        tmp_pop_p = nest.Create("diff_neuron_nestml", n=numNeurons, params=par_fbk)
        nest.SetStatus(
            tmp_pop_p,
            {
                "pos": True,
                "buffer_size": buf_sz,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.fbk_p = PopView(tmp_pop_p, to_file=True, label="mc_fbk_p")

        tmp_pop_n = nest.Create("diff_neuron_nestml", n=numNeurons, params=par_fbk)
        nest.SetStatus(
            tmp_pop_n,
            {
                "pos": False,
                "buffer_size": buf_sz,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.fbk_n = PopView(tmp_pop_n, to_file=True, label="mc_fbk_n")

        ############ OUTPUT ############
        tmp_pop_p = nest.Create("basic_neuron_nestml", n=numNeurons, params=par_out)
        nest.SetStatus(
            tmp_pop_p,
            {
                "pos": True,
                "buffer_size": buf_sz,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.out_p = PopView(tmp_pop_p, to_file=True, label="mc_out_p")

        tmp_pop_n = nest.Create("basic_neuron_nestml", n=numNeurons, params=par_out)
        nest.SetStatus(
            tmp_pop_n,
            {
                "pos": False,
                "buffer_size": buf_sz,
                "simulation_steps": len(self.motorCommands),
            },
        )
        self.out_n = PopView(tmp_pop_n, to_file=True, label="mc_out_n")

        nest.Connect(
            self.m1_out_p.pop,
            self.out_p.pop,
            "one_to_one",
            {"weight": params.wgt_ffwd_out},
        )
        nest.Connect(
            self.m1_out_p.pop,
            self.out_n.pop,
            "one_to_one",
            {"weight": params.wgt_ffwd_out},
        )
        nest.Connect(
            self.m1_out_n.pop,
            self.out_p.pop,
            "one_to_one",
            {"weight": -params.wgt_ffwd_out},
        )
        nest.Connect(
            self.m1_out_n.pop,
            self.out_n.pop,
            "one_to_one",
            {"weight": -params.wgt_ffwd_out},
        )

        nest.Connect(
            self.fbk_p.pop,
            self.out_p.pop,
            "one_to_one",
            {"weight": params.wgt_fbk_out},
        )
        nest.Connect(
            self.fbk_p.pop,
            self.out_n.pop,
            "one_to_one",
            {"weight": params.wgt_fbk_out},
        )
        nest.Connect(
            self.fbk_n.pop,
            self.out_p.pop,
            "one_to_one",
            {"weight": -params.wgt_fbk_out},
        )
        nest.Connect(
            self.fbk_n.pop,
            self.out_n.pop,
            "one_to_one",
            {"weight": -params.wgt_fbk_out},
        )

    def connect(self, planner_p, planner_n):
        self.m1.connect((planner_p, planner_n))
