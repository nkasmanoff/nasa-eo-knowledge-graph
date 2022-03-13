# nasa-eo-knowledge-graph


This repository houses the code for a very quick demonstration on how to use the captions collected from NASA Earth Observatory's natural event curation to construct a knowledge graph,
relating entities with relations based on the Spacy extracted grammatical properties of each. 

This work is heavily based https://www.kaggle.com/nageshsingh/build-knowledge-graph-using-python, and https://github.com/lingfeiwu/people2vec/blob/master/TransE_WorldCup2014.ipynb, meant to get the code started, but there are plenty of ways to build on this work noted below. 


## Early Results

The following is one extracted portion of the full knowledge graph, where we can see what entities are related by the verb "triggered"

![](https://github.com/nkasmanoff/nasa-eo-knowledge-graph/bin/entities linked by triggered.png) 




## Future Steps / Questions
    - would an Earth Science Specific text model be able to better extract entities and relations?
    - would using the entirety of the NASA EO dataset make for richer data?
    - can we group nodes and relations into specific categories? (i.e event, location, causes, affects, etc.)
    - can the embeddings obtained from TransE or a similar algorithm facilitate the use of GNNs? 
    - Is there a way to introduce images into this work? 
