# Change the parameters ONLY in run_something.py
If you only running the code and want to change any parameter, run it in run_something.py with ready template.
This is needed to make sure that the logic code changes will not be changed unintentionally and so merging confict apears.

IF the files was accidently changed and merging confict apears, delete the file and pull it from the repo, then push.

# scape_sem_o.py
THis is a data scraper for the SEM-O datasource. Currently it outputs the daily results for the past 90 days period.
The report name is displayed in the link of report's page. For example for report "https://www.sem-o.com/market-data/dynamic-reports#BM-086" the report title is "BM-086".
To change the report for scraping - check 

# graph_lib.py
Uses the previous techniques with geomaps coordinates and averaging and then avaraging the results even one more time.
As no generators in generators.csv are with known coordinates, their position is estimated as random position near the closest connected bus. 
This module is reused in get_weighted_graph.py, betweenness.py, spectrum.py, cluster.py

# graph_lib.py -> betweeness.py, spectrum.py -> cluster.py
These are the construction and analysis of the weighted graph (with or without generators). The weights are purely the power capasity of the lines. 

Running the code is reccomended for transmission graph only as it is more valid and reccomended by the report in github repo.

Betweeness calculates the centality (how the notes are connected to other).

Clustering is performed on the eigenvectors of the Laplacian matrix using k-mean clustering approach.

# bus_connection.py
Purly statistical buses analysis on how the grid is constructed.
Outputs the histograms for reference.

# /flow
Some attempt to perform analysis of the grid with PyPsa flow simulation. Similar to get_weighted_graph.py pipeline but for different time snapshots over syntetic data.

# /flow/upgrade_lines.py
The idea of iterative grid updading. Needs more work in it.

# Note
These approaches are highly AI generated and needs a manual check. This is only made for demostrational purposes.