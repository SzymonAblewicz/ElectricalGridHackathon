# scape_sem_o.py
THis is a data scraper for the SEM-O datasource. Currently it outputs the daily results for the past 90 days period.
The report name is displayed in the link of report's page. For example for report "https://www.sem-o.com/market-data/dynamic-reports#BM-086" the report title is "BM-086".
To change the report for scraping - check 

# betweeness.py, cluster.py, get_weighted_graph.py 

These are the construction and analysis of the weighted graph (with or without generators). The weights are purely the power capasity of the lines. 

Running the code is reccomended for transmission graph only as it is more valid and reccomended by the report in github repo.

Betweeness calculates the centality (how the notes are connected to other).

Clustering is performed on the eigenvectors of the Laplacian matrix using k-mean clustering approach.


These approaches are highly AI generated and needs a manual check. This is only made for demostrational purposes.