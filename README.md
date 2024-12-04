# Docker build and run
```
docker build -t graphmatch-clone .
docker run -i -t graphmatch-clone /bin/bash
```

# Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree
Code for paper "Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree", SANER 2020  
Requires:   
pytorch    
javalang  
pytorch-geometric  

## Data
Google Code Jam snippets in googlejam4_src.zip  
Google Code Jam clone pairs in javadata.zip  
BigCloneBench snippets and clone pairs in BCB.zip  
