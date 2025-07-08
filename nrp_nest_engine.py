"""Python Engine 1. Will get current engine time and make it accessible as a datapack"""

from nrp_core.engines.python_json import EngineScript
import random

from complete_control.neural.NestClient import NESTClient

NANO_SEC = 1e-9

nest = NESTClient(host='nest-server', port=9000)

class Script(EngineScript):
    def initialize(self):
        """Initialize datapack1 with time"""
        print("Engine 1 is initializing. Registering datapack...")
        self._registerDataPack("spkRate")
        self._setDataPack("spkRate", {"rate": 0.0 })

        print("attempting nest call")
        print(nest.GetKernelStatus())
        self.neuron = nest.Create("iaf_psc_alpha")
        nest.SetStatus(self.neuron, {"I_e":376.0})
        self.neuron2 = nest.Create("iaf_cond_alpha")
        self.rec = nest.Create("spike_recorder")
        print(f"recorder creation call returned {self.rec}")
        self.syn = nest.Connect(self.neuron, self.neuron2)
        print("synapse return from connect call: ", self.syn)
        print("getstatus on synapse", nest.GetStatus(self.neuron))
        nest.Connect(self.neuron, self.rec)
        
        print("Finished initializing")

    def runLoop(self, timestep_ns):
        """Update spkRate at every timestep"""
        nest.Simulate(timestep_ns * NANO_SEC)
        res = nest.GetStatus(self.rec)[0]
        # res = self.rec.GetStatus()[0]
        print("res: ", res)
        ts = res["events"]["times"]
        self._setDataPack("spkRate", {"rate":len(ts)})
        print("spkRate data is " + str(self._getDataPack("spkRate")))

    def shutdown(self):
        print("Engine 1 is shutting down")

    def reset(self):
        print("Engine 1 is resetting")
