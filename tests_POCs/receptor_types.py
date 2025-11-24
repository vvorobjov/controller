"""
Demonstrates weird NEST behavior with receptor types.
Default receptor is receptor 0, but if you create a neuron with custom receptors, then the first receptor (which you created with receptor: 1) also has ID 0.
"""

import nest

nest.Install("custom_stdp_module")
a = nest.Create("iaf_cond_alpha")
pc = nest.Create("eglif_pc_nestml")

print("eglif_pc_nestml", nest.GetDefaults("eglif_pc_nestml")["receptor_types"])
nest.Connect(a, pc, "one_to_one", {"receptor_type": 1})
print("eglif_pc_nestml ", nest.GetStatus(nest.GetConnections(a, pc)[0], "receptor"))


# pc2 = nest.Create("eglif_pc_nestml")
# b = nest.Create("iaf_cond_alpha")
# print("eglif_pc_nestml ", nest.GetStatus(nest.GetConnections(a, pc2)[0], "receptor"))
# print("iaf_cond_alpha ", nest.GetStatus(nest.GetConnections(a, b)[0], "receptor"))
# nest.Connect(a, pc2, "one_to_one", {"receptor_type": 4})
# nest.Connect(a, b, "one_to_one")
