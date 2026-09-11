a
for x in range(1):
	a=1
	b=2
exit()
while 1:
	print("wazaaaaaaaaaaaaaap ", end=' ')
exit()
for
exit()
a
exit()
27+64+125
exit()
exit
:q
exit)_
exit()
:q
exit()
while (1):
	a=1
exit()
2**32
2**21
2**31
2**64
exit()
print("x")
a
b
c
import math
math
math.x
math.choose(3,3)
math.comb(3,3)
exit()
import pypsa
exit()
import pypsa
n = pypsa.Network()
from numpy import pi
n.add("Bus", ["zone_1", "zone_2", "zone_3"], v_nom=11)
n.add(
    "Load",
    ["load_1", "load_2", "load_3"],
    bus=["zone_1", "zone_2", "zone_3"],
    p_set=[50, 60, 300],
)
n.add(
    "Generator",
    ["gen_A", "gen_B", "gen_C", "gen_D"],
    bus=["zone_1", "zone_1", "zone_2", "zone_3"],
    p_nom=[140, 285, 90, 85],
    marginal_cost=[7.5, 6, 14, 10],
)
n.add(
    "Line",
    ["line_1", "line_2", "line_3"],
    bus0=["zone_1", "zone_1", "zone_2"],
    bus1=["zone_2", "zone_3", "zone_3"],
    s_nom=[126, 250, 130],
    x=[0.02, 0.02, 0.01],
    r=0.01,
)
n.optimize()
n.model
display(n)
display()
display("a")
n.model.constraints["Bus-nodal_balance"]
display(n.buses_t.marginal_price)
display(n.generators_t.p)
display(n.lines_t.p0)
from IPython.display import display
display(n)
display(n.buses_t.marginal_price)
n.plot
display(n.plot)
n.plot(margin=0.1)
display()
plot(n)
n.plot
n_import = pypsa.Network("participant-kit/networks/SV2033_all-island.nc")
display(n_import)
n_import.plot()
matplotlib
import matplotlib.pyplot
matplotlib.pyplot.show()
n_import
n_import.buses
n_import.lines
for i in n_import.lines:
	print(i.bus0)
	print(i)
for i in n_import.lines:
	print(i)
n_import.lines
n_import.lines.bus0
n_import.lines.buss_nom.max
n_import.lines.buss_nom.max()
n_import.lines.buss_nom
n_import.lines.bus.s_nom
n_import.lines.snom
n_import.lines.s_nom
n_import.lines.s_nom.max
n_import.lines.s_nom.sorted()
n_import.lines.s_nom.sorted
n_import.lines.s_nom
n_import.lines.s_nom.is_monotonic_decreasing
n_import.lines.s_nom.unique()
n_import.lines.s_nom.values
n_import.lines.s_nom.values.sorted
n_import.lines.s_nom.values.sorted()
n_import.lines.s_nom.values.sort
n_import.lines.s_nom.values.sort()
a = n_import.lines.s_nom.values.copy()
a.sort()
a
b = n_import.copy()
diff(a,b)
b == n
b == n.copy
b == n_import
b.optimize()
b == n_import
b
b == n_import
c = n_import.copy()
c == n_import
c == b
b
c
n_import
n_import.Lines
n_import.lines
n_import.lines.loc
clear
clear()
n_import.lines
n_import.lines_t
n_import.lines_t.p0
n_import.lines_t.p1
n_import.lines.p0
b.lines.p0
b.lines_t.p1
d = n_import.copy()
d.lines_t
d.lpf()
d.lines_t.p0
n_import
n_import.generators
n_import.lines_t
n_import
n_import.generators_t
d.generators.pset
d.generators.p_set
n_import.generators.p_set
n_import.generators_t.pset
n_import.generators_t.p_set
d.generators_t.p_set
d.generators_t
d.generators_t.p1
d.generators_t.p_nom
d
e
d
d.lines_t
d.lines_t.p0
d.buses_t
d.generators_t
d.generators_t.p
d.generators_t.q
b.generators_t.p
d.lines_t.p0
b.lines_t.p0
n_import.generators
tmp = d.generators_t.p.copy()
tmp.sort()
tmp
type(tmp)
zero_cols = tmp.columns[(df == 0).all(axis=0)]
zero_cols = tmp.columns[(tmp == 0).all(axis=0)]
zero_cols
tmp.drop(columns=zero_cols, inplace=True)
tmp
n_import.col("52071-1")
n_import.generators["52071-1"]
n_import.generators
n_import.generators[n_import.generators["name"] == "10470-1"]
n_import.generators[n_import.generators["bus"] == "10470-1"]
n_import.generators.loc["10470-1"]
n_import.generators.loc["52071-1"]
gh = n_import.copy()
gh.calculate_PTDF()
gh.subnetworks
gh.sub_networks
subnet = gh.sub_networks.obj["0"]
subnet
subnet = gh.sub_networks.obj["1"]
subnet
subnet = gh.sub_networks.obj["2"]
subnet
gh.sub_networks
subnet = gh.sub_networks.slack_bus["1021"]
subnet = gh.sub_networks.slack_bus[1021]
subnet = gh.sub_networks.slack_bus
subnet
subnet = gh.sub_networks.slack_bus[0]
subnet = gh.sub_networks.slack_bus["1021"]
subnet
subnet = gh.sub_networks.loc["1021"]
gh.sub_networks.loc["1021"]
gh.sub_networks.loc["0"]
subnet = gh.sub_networks.loc["0"]
gh
gh.determine_network_topology()
subnet = gh.sub_networks.loc["0"]
pypsa.__version__
gh
print(gh.sub_networks)
sn = gh.sub_networks.obj["0"]
sn.calculate_PTDF()
sn.PTDF
import pandas as pd
ptdf = pandas.DataFrame(
	sn.PTDF,
	index = sn.branches_i(),
	columns = sn.buses_o,
)
ptdf = pd.DataFrame(
	sn.PTDF,
	index = sn.branches_i(),
	columns = sn.buses_o,
)
ptdf
n_import.lines.v_norm
n_import.lines.v_ang_max
n_import.lines.v_ang_min
b
b.buses.v_nom
print(v)
print(a)
a
type(a)
import PIL
PIL
ls
import os
os.sys("ls")
os.listdir()
os.read(eirgrid.py)
for x in range(300)
for x in range(300):
	print(" ")
clrscr()
clear_screen()
ptdf
sn.generators
print(sn.generators)
sn.generators[0]
n_import.generators
n_import.transformers
n_import.generators.at["10470-1","bus"]
ptdf
ptdf["10422"]
ghf = ptdf["10422"]
ls
ghf
ghf.sort_values()
import matplotlib.pyplot as plt
plt.hist(ghf)
plt.show()
plt.hist(ghf,bins=60)
plt.show()
plt.hist(ghf,bins=100)
plt.show()
print(ghf)
print (ghf.values)
connected_lines = n.lines[(n.lines['bus0'] == target_bus) | (n.lines['bus1'] == target_bus)]
connected_lines = n_import.lines[(n_import.lines['bus0'] == "10422") | (n.lines['bus1'] == "10422")]
len(connected_lines)
connected_lines = n_import.lines[(n_import.lines['bus0'] == "10422") | (n_import.lines['bus1'] == "10422")]
connected_lines
len(connected_lines)
guy = []
for i in n_import.buses:
	guy.append( n_import.lines[(n_import.lines['bus0'] == i) | (n_import.lines['bus1'] == i)])
guy
n_import.buses
for i in n_import.buses:
	print(i)
for i in n_import.buses.names:
	print(i)
n_import.buses.items
n_import.buses.items[1]
n_import.buses.index
guy = []
for i in n_import.buses.index:
	guy.append( n_import.lines[(n_import.lines['bus0'] == i) | (n_import.lines['bus1'] == i)])
guy
len(guy)
guy = []
for i in n_import.buses.index:
	guy.append(len( n_import.lines[(n_import.lines['bus0'] == i) | (n_import.lines['bus1'] == i)]))
guy
plt.hist(guy)
plt.show()
binsize = 0.5
bin_edges = np.arange(start=-1.0, stop=10 + bin_size, step=bin_size)
import numpy as np
bin_edges = np.arange(start=-1.0, stop=10 + bin_size, step=bin_size)
bin_edges = np.arange(start=-1.0, stop=10 + binsize, step=bin_size)
bin_edges = np.arange(start=-1.0, stop=10 + binsize, step=binsize)
plt.hist(guy, bins = bin_edges)
plot.show()
plt.show()
n_import.graph()
plt.show()
display(n_import.graph())
sum(guy)
n_import.plot()
plt.show()
n_import.plot(bus_sizes=0.01)
plt.show()
n_import.plot(bus_sizes=0.001)
plt.show()
n_import.plot(bus_sizes=0.0001)
plt.show()
n_import
guy
sum(guy)
sum(guy)/2
import readline
readline.write_history_file("pypsa_hist.py")
