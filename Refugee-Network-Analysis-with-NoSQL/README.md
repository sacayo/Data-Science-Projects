# Refugee Network Analysis using NoSQL

## Collaborator
[Eric Ha](https://github.com/awesome-eric)\
[Chloe McGlynn](https://github.com/camcglynn)\
[Melody Lamphear](https://github.com/melodymassis)

## Objective
This project aimed to analyze and visualize global refugee movements, focusing on the question: "Where are most refugees coming from/moving to, and how many were turned away from country of asylum during recent years (2020-2022)?" We sought to leverage NoSQL database technology to gain insights into refugee patterns and identify influential countries in the global refugee network.

## Methodology
Our approach involved several key steps:

##### Data Integration:

Combined data from three major sources: UNHCR, UNRWA, and IDMC
Utilized the R package {refugees} which includes 8 datasets


##### Data Processing and Exploration:

Used SQL for initial data exploration and preprocessing
Created query_combined tables with different options for analysis


##### Network Analysis with Neo4j:

Constructed graph representations of refugee movements
Countries represented as nodes, refugee flows as directed relationships
Created two main graph models:
a. Inclusive of NULL return numbers
b. Removed rows with no Returned numbers


##### Advanced Analytics:

Applied PageRank and Personalized PageRank algorithms to identify influential nodes
Utilized Louvain Modularity for community detection within the refugee network


##### Visualization:

Created network visualizations using Neo4j
Developed map-based visualizations using Google Maps API



## Tools and Technologies:

R for data processing and the {refugees} package\
SQL for data exploration\
Neo4j for graph database and network analysis\
Google Maps API for geographical visualizations

## Results
Key findings from our analysis include:

Dramatic increase in refugees due to war over 2020-2022, with Syria, Afghanistan, and Ukraine creating the most refugees
Refugee count reached a 10-year high in 2022 (~30M) compared to the prior decade average of ~15M
Identified regional patterns of migration:

African refugees mostly settle in Africa\
Refugees from other regions primarily move to Europe\
Exception: China's primary settlement country is the US


Network analysis revealed key transit countries and refugee hubs

## Significance/Impact
This project provides valuable insights into global refugee movements and demonstrates the power of graph databases in analyzing complex social phenomena:

### Policy Implications: 
The findings can inform international policy on refugee support and asylum processes.\
Humanitarian Aid: Insights into refugee flows can help organizations better allocate resources and plan aid distribution.\
Predictive Capabilities: The network analysis lays groundwork for predictive models of future refugee movements.\
Technological Application: Demonstrates the effectiveness of NoSQL databases, particularly graph databases, in handling and analyzing complex, interconnected data.

### Future Directions and Applications
The project opened up several avenues for future work and practical applications:

Storing precomputed paths for refugee movement based on historical data\
Developing heuristics from identified migration patterns\
Creating a web server for a Refugee Data Visualization Dashboard\
Implementing a front-end cache for refugee data visualization\
Managing user preferences and customizations for personalized analysis

This project showcases the application of advanced data engineering and analysis techniques to a critical global issue, demonstrating the ability to derive meaningful insights from complex, multi-source datasets while leveraging cutting-edge NoSQL database technology.
