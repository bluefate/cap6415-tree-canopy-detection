# Tree Canopy Detection

Project for the Solafune Tree Canopy Detection competition.

## Summary

This repo contains code and notebooks to train and evaluate models that segment tree canopy from RGB TIFF aerial and satellite imagery. The competition uses polygon-based JSON annotations. Submissions are a single JSON file in the prescribed schema.

Source summary from the competition page: 
- images are RGB TIFFs.
- Training annotations contain polygon segmentations with class and confidence fields.
- Submissions must be one JSON file that matches the sample format.
- See the competition overview for details.

## Notes from Discussion Board
- Must use DockerFile to provide environment used
- If using NVIDIA’s GPUs, please ensure your environment supports CUDA 11.8 or above.

## URL
'https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70'

## Folders - BASH
`mkdir -p {data/{raw/{images,annotations},processed/{tiles,masks},splits,sample_submission},notebooks,models,src/{data,models,scripts,utils,submission},submission}`


## Repository layout
