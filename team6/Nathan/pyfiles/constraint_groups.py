#!/usr/bin/python3
#takes command line arg of .nc file corresponding to network

import sys, pypsa
import matplotlib.pyplot as plt
import scipy
import numpy

n = pypsa.Network(sys.argv[1])

n.determine_network_topology()

ptdf_cutoff = 0.1

subnets = len(n.sub_networks.obj)

load_cutoff = 0.9

# def constraint_groups(subnetwork: pypsa.SubNetwork, outlist):

i=n.sub_networks.obj["0"]
print(i)
i.calculate_PTDF()
the_matrix = i.PTDF

change_max = 20

b_vec = numpy.ones(shape=(10,1))
power_change_bound_upper = n.lines.s_nom.to_numpy() - numpy.abs(n.lines_t.p0.iloc[0,:].to_numpy())
power_change_bound_lower = numpy.minimum(tmp_size := numpy.abs(n.lines_t.p0.iloc[0,:].to_numpy()),change_max*numpy.ones(shape = (1,len(tmp_size))))
