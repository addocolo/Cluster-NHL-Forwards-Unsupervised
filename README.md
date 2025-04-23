# Clustering NHL Forwards by Game Statistics

Final project for Unsupervised Learning

In this study we will use unsupervised learning techniques to attempt to cluster ice hockey forwards from the National Hockey League into groups based on their in-game statistics from the 2023-24 season. We will analyze the statistics of each group's players and assign a label and description to the players. By using unsupervised learning we hope to uncover natural groupings of players based on their statistics.

The data set used is from the website Natural Stat Trick, a site dedicated to statistical analysis of NHL players and teams. The data was retrieved in csv format and a [link](https://www.naturalstattrick.com/playerteams.php?fromseason=20232024&thruseason=20232024&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL) to the data can be found in the Reference section. The references also contain a link to a GitHub repository for this project.

The raw data contains 35 features. Not all of these features will be used for our analysis, however so we will save a description of the relevant features for our Data Cleaning section. The 35 features in the data source are 'Unnamed: 0, Player, Team, Position, GP, TOI, Goals, Total Assists, First Assists, Second Assists, Total Points, IPP, Shots, SH%, ixG, iCF, iFF, iSCF, iHDCF, Rush Attempts, Rebounds Created, PIM, Total Penalties, Minor, Major, Misconduct, Penalties Drawn, Giveaways, Takeaways, Hits, Hits Taken, Shots Blocked, Faceoffs Won, Faceoffs Lost, and Faceoffs %'.

## Data Import and Cleaning
From an initial inspection of the data, we can see that it contains 924 player entries with 35 features each. Excluding the index, Player, Team, and Position columns, which are there to identify the player, there are 31 numeric columns we can use to cluster players.

These 31 features contain a number of statistics, both traditional and some more advanced statistics. Although we will not use all of the features in our final model, we will clean and prepare them all for analysis so that we can easily add or remove them from our model later. Initial cleaning steps included replacing null values with 0 and removing players with fewer than 20 games played and removing players who play non-forward position (Defensemen).

We remove some features from consideration during training. Some of these statistics are redundant, for example Total_Assists = First_Assists + Second Assists and Total_Points = Goals + Total_Assists. Penalty time is accounted for in PIM (Penalties in Minutes), so we removed Total_Penalties, Minor, Major, and Misconduct which all measure different ways a player can be penalized. Since mainly players playing Center take faceoffs, all of the faceoff categories were likely to influence the model to cluster by position rather than player type, so they were removed. Games played was also removed since TOI (Time on ice) seemed a related but better measure of playing time. Other features were removed after some testing of different features sets. While the advanced metrics IPP, iFF, iSCF, ixG, iCF, and iHDCF were all included in initial trials, it was discovered that including these seemed introduce more confusion into the clusters rather than helping. This may be due to having high collinearity with each other. Giveaways and Shooting Percentage didn't seem to add anything to the clusters as their means tended to be similar across all clusters regardless of what K was set for KMeans.

After cleaning our data we are left with the following features that we will use to cluster players:

- **TOI**: *Total time on ice; the amount of playing time in minutes for each player.*  

- **Goals60**: *Goals scored per 60 minutes; a normalized goal metric.*  
- **First_Assists60**: *Primary assists per 60 minutes; assists where the player is the first to set up a goal.*  
- **Second_Assists60**: *Secondary assists per 60 minutes; assists where the player contributes secondarily to a goal.*  
- **Shots60**: *Shots taken per 60 minutes.*  
- **Rush_Attempts60**: *Rush or fast-break attempts per 60 minutes, indicating aggressive offensive actions.*  
- **Rebounds_Created60**: *Rebounds generated per 60 minutes, indicating high shot volume.*  
- **PIM60**: *Penalty minutes per 60 minutes; the average time a player spends serving penalties.*  
- **Penalties_Drawn60**: *Penalties drawn per 60 minutes; an indication of how often a player forces opponents into penalties.*  
- **Takeaways60**: *Takeaways per 60 minutes; instances of stealing possession from the opposing team.*  
- **Hits60**: *Hits delivered per 60 minutes.*  
- **Hits_Taken60**: *Hits received per 60 minutes.*  
- **Shots_Blocked60**: *Opponent shots blocked per 60 minutes.*

## EDA
We can see that our data features vary significantly in scale and distribution, with large gaps between minimum, maximum, and mean values. As such we use a standard scaler to scale all of our features around 0 and scale the data to unit variance rendering it more suitable for distance-based clustering in higher dimensions. This should ensure that no one feature unduly dominates the others when clustering.

![image](https://github.com/user-attachments/assets/309f88e9-4531-4504-a3eb-7af269293bc6)

Examining a boxplot of scaled features indicates that all metrics are now standardized around similar ranges. There are some outliers in certain categories, most notably PIM60, however we opted to retain these as some represent rare but valid player types (e.g., enforcers—players used more for physical intimidation than offensive or defensive prowess).

![image](https://github.com/user-attachments/assets/5476a7a6-fb4d-4c9f-b0dc-9f5b6b24bbd3)

From the features we selected, there does not appear to be any high degree of collinearity. Some features show a moderate degree of correlation, but all of these correlated features are <= 0.71 and represent different, though related things. For example, the greatest correlation is between Shots60 and Goals60 (0.71). It stands to reason that more shots will likely result in more goals, but higher goals in fewer shots may represent a more talented shooter.

![image](https://github.com/user-attachments/assets/354cc58b-02f3-4428-9979-fb7546fd5185)

This is supported by a pairwise plot of all features which suggests some mild correlations at most. This low collinearity supports our use of all selected features without needing dimensionality reduction for redundancy.

![image](https://github.com/user-attachments/assets/d7603222-3d95-43a3-916f-2e264102a88d)

We applied a t-SNE (t-Distributed Stochastic Neighbor Embedding) to attempt to visualize our high dimension data in two dimensions to see if there is any evident cluster pattern. This dimensionality reduction technique attempts to preserve higher dimensional distances in a 2 dimensional space and can potentially reveal groups or patterns in our higher dimensional data. Unfortunately in this case no obvious pattern emerges just yet.

![image](https://github.com/user-attachments/assets/afe307f1-1bdd-40f2-9fde-4fb459b21fed)

## KMeans Clustering

KMeans was selected as the primary model type because it minimizes within-cluster variance, making it effective for grouping players in higher-dimensional space. After clustering, we will examine the characteristics of each group to understand what player types they represent.

### Hyperparameter Tuning

Since we will be using a KMeans model to cluster our data, we used KElbowVisualizer. This runs KMeans with multiple values of K and attempts to determine the ideal value of K, where increasing the number of clusters doesn't significantly improve performance. In our case, we tested values of K from 3 to 10. The best value of K in this range was 5.

![image](https://github.com/user-attachments/assets/6dedabb3-a949-4e79-8ae8-30648ffca2e8)

### Model

Using the K or n_clusters value identified by our K-Elbow, we used sklearn's KMeans to cluster our player data.

![image](https://github.com/user-attachments/assets/736b79c0-50cf-4997-a473-3a5b609fb0cc)

### Analysis

A bar plot of the number of players in each cluster shows us an uneven distribution of players with the largest cluster, cluster 4, containing just under 175 players and the smallest, cluster 3, with only about 35. This is to be expected for clustering highly complex sports statistics. Grouping players by types, there are without a doubt some categories that would be much smaller than others. Cluster 4 may represent a broad category of players with average statistics whereas cluster 3 may identify a more niche player type with a specialized role.

![image](https://github.com/user-attachments/assets/7243486d-a984-415d-ad9e-81a35fdfb179)

We re-examined our t-SNE from above, but with the clusters color-coded. While there is some overlap in the 2 dimensional representation, the clusters do seem to be moderately well-defined in this space, suggesting that KMeans was able to pick up on some underlying patterns in the feature space.

![image](https://github.com/user-attachments/assets/bda91dcf-daf4-49eb-bee3-9e3732d7543a)

In order to assign meaningful labels to our clusters, we examined the distribution of each feature across each cluster. By analyzing these distributions, we can begin to understand which player statistics differentiate the clusters, enabling us to assign meaningful labels in the following results section.

![image](https://github.com/user-attachments/assets/e4f966ef-8f54-481a-b8e3-e5941f5135e6)

### Results

Our K-elbow analysis indicated that five clusters are optimal within our test range. The KMeans model separated the data into five distinct clusters. By examining the distribution of key statistics across these clusters, we assigned each a meaningful verbal label along with a short description that highlights the traits driving its separation.

- Cluster 0 – Defensive Specialists:
These players show lower offensive outputs—such as lower goals, assists, and scoring chance metrics—compared to the other clusters. However, their higher rates of shot-blocking suggest a focus on defensive responsibilities. This cluster likely represents players who focus on protecting the defensive zone and limiting opponents’ opportunities.

- Cluster 1 – Shooters:
This group is marked by the highest counts in shots, goals, and rebounds. Their statistical profile indicates a primary focus on scoring, with an aggressive approach in front of the net. These players are clearly geared toward generating offensive opportunities and finishing plays.

- Cluster 2 – Playmakers:
Characterized by standout assist numbers, the players in this cluster lean toward setting up their teammates rather than scoring themselves. Their elevated assist totals underscore a creative role focused on distribution rather than finishing plays themselves.

- Cluster 3 – Physical Forwards/Enforcers:
Distinguished by noticeably lower offensive outputs (both goals and assists), this group compensates with elevated physicality, as evidenced by higher penalty minutes (PIM) and a higher rate of hits. These players embrace a tougher, more physical style of play—playing the role of enforcers or checking specialists who are relied upon for their grit and toughness.

- Cluster 4 – Two-Way Forwards:
Representing the largest group, these players exhibit balanced statistical characteristics. They do not stand out in any single offensive or defensive area but contribute steadily across both ends of the ice. This versatile profile suggests that they are relied upon to perform in multiple roles, acting as consistent and adaptable forwards.

## Other Models
In this section, we explore alternate clustering methodologies to evaluate the robustness of our clustering strategy. By comparing methods, we can assess whether the groupings we see with KMeans are consistent when alternative techniques or dimensionality reduction are used.

### Agglomerative Clustering

For consistency with our KMeans setup, we use the same value of n_clusters when applying Agglomerative Clustering. We used the Ward linkage method, which seeks to minimize the within-cluster variance in a hierarchical fashion.

### PCA + Clustering

Since our datasets are high-dimensional, we implement PCA as a way to reduce the number of variables while retaining at least 90% of the original variance. After applying PCA, we perform clustering in the reduced space using two techniques:

- PCA + KMeans: The PCA-transformed data is clustered using KMeans, ensuring that the dimensionality reduction doesn't substantially alter our clustering structure.

- PCA + Agglomerative Clustering: We also run Agglomerative Clustering on the PCA-reduced data. This allows us to observe if similar groupings occur when using hierarchical approaches on a more compact feature set.

### Analysis
#### Cluster Alignment

Before comparing the results of the different clustering techniques, we need to align their resulting clusters. This is because the clusters labels (0.., n_clusters) are assigned arbitrarily so KMeans cluster 1 could theoretically match perfectly with AggClust 3. We optimize the matching of the clusters using the Hungarian algorithm.

#### Measuring pairwise similarity

We checked the similarity between our primary KMeans clustering assignment and each of the other clustering methods attempted. Since there are no ground truth labels in our project, we used the Normalized Mutual Information (NMI) metric to evaluate similarity between two clusters. NMI measures the mutual dependence between clustering assignments, ensuring that the results are comparable even in the absence of predefined class labels. A high NMI score indicates strong alignment between clustering assignments, meaning that different clustering methods produce similar clusters.

![image](https://github.com/user-attachments/assets/cd6929cf-ec40-40bf-bf74-0c142f5ab00d)

## Conclusion

<span style='font-size: 24px;'>***KMeans Clustering Model***</span>

In this project, we applied unsupervised learning techniques to cluster NHL forwards from the 2023–24 season based on a wide array of in-game performance metrics. Our primary model, KMeans clustering, identified five distinct player types:

  - Defensive Specialists, who prioritize blocking shots and limiting opponents' scoring chances,

  - Shooters, who generate high shot volume and goal totals,

  - Playmakers, who lead in assists and excel at setting up teammates,

  - Physical Forwards/Enforcers, who contribute less offensively but bring high physical intensity,

  - Two-Way Forwards, who show a well-rounded balance across all areas.

These clusters align well with intuitive playing styles seen in professional hockey and provide interpretable, role-based groupings that coaches, analysts, or scouts might find useful.

<span style='font-size: 24px;'>***Other Clustering Models***</span>

  To evaluate the robustness of our findings, we compared KMeans to Agglomerative Clustering and two PCA-based variants. Using Normalized Mutual Information (NMI) as our similarity metric, we found that the PCA-based KMeans clustering most closely aligned with our primary model (NMI = 0.60), while Agglomerative Clustering showed slightly lower agreement (NMI ≈ 0.57), both with and without PCA.

  These modest differences suggest that while the core clustering structure keeps some consistency across models, edge cases may be grouped differently depending on the clustering algorithm used. Additionally the PCA transformation somewhat alters the geometry of the data, which seems to slightly affect cluster boundaries. The observed variation in these models signifies that different algorithms may emphasize different aspects of player behavior.

<span style='font-size: 24px;'>***Limitations***</span>

  While our clustering analysis yielded interpretable player groupings, there are several important caveats and constraints to keep in mind:

  1. **Single‐Season, Forward‐Only Scope**: We used only one season of NHL data (2023–24) and restricted our analysis to forwards. Player performance can vary substantially from year to year so clusters found here may not generalize to other seasons or to defensemen.

  2. **Model Selection**: We limited our study to K-Means and Agglomerative Clustering, the methods with which we’re most familiar. Other algorithms (e.g., density-based, spectral clustering) might uncover more nuanced groupings or better capture rare player archetypes.

  3. **Lack of External Validation**: Clusters were not cross-checked against expert labels or coaching roles. Therefore, there’s no guarantee that every player in a given cluster would be recognized as that “type” by hockey analysts or scouts.

<span style='font-size: 24px;'>***Future Research***</span>

  In terms of future study to extend this project, there are a number of potentially interesting avenues:

  1. **Temporal analysis**: Track how player cluster assignments change over time, either within a season or across multiple seasons.

  2. **Incorporate goalie and defensemen data**: For this study we limited the scope to forwards, but it would be interesting to expand the analysis to include other positions and understand how their playing styles cluster differently.

  3. **Role projection for prospects**: Apply the clustering model to junior league or lower league players to project potential NHL roles based on similar statistical profiles.

## References

Dataset:
"Player Season Totals". *Natural Stat Trick*. 2024. https://www.naturalstattrick.com/playerteams.php?fromseason=20232024&thruseason=20232024&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL
