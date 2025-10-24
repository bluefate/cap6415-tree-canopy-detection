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
- The YOLO series model from Ultralytics is allowed to be used in the competition
- If using NVIDIA's GPUs, please ensure your environment supports CUDA 11.8 or above.
- You do not need to train a classification model for scene_type or cm_resolution. These fields are used only as weights in the evaluation. For what to include in your submission, follow the sample submission in the starter materials.
  1. scene_type and cm_resolution are not prediction targets for participants.
  2. They are internal weighting factors applied during evaluation.
  3. When preparing your file, match the columns and formats exactly as shown in the sample submission (see the Starter Notebook / sample submission for reference).
  - If you're unsure which variables must be present in your file, copy the structure from the sample submission.
- **We allow the use of software under Ultralytics licenses, but as a general rule, we prohibit the use of software under GPL or AGPL licenses since they are copyleft licenses.**
    - all Ultralytics YOLO series models are allowed for use in this competition
- If you're using a semantic segmentation model instead of instance segmentation. If that's the case then you'd need to either basically start from scratch with an instance segmentation approach or you could post-process your results using something like watershed to split up the mask (try computing a binary distance map, maybe apply Gaussian smoothing, then find local peaks, and use those peaks to seed the watershed algorithm).

1. Please do not upload the raw dataset itself to external AI services such as ChatGPT or other cloud assistants. Doing so would be considered releasing the data, which is prohibited under the rules.
2. Using an AI assistant for code generation and review is allowed in this competition. However, please note that publishing the solution you use for your final submission on GitHub or any similar platform during the competition period would clearly constitute a violation of the rules. The allowance for AI assistant usage may be subject to change in future competitions.

- [Data Dictionary](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=discussion&tab=&id=&topicId=fb298f0f-7ca5-426a-9fd8-c41efa0de87e)
- [Converting to COCO format annotation](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?tab=&menu=discussion&page=2&topicId=d7a7a13d-e8e7-489d-b8d9-7dca36ae30b7)
- [Submission preparation code](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=discussion&tab=&id=&page=2&topicId=05a2491b-2094-4f9d-b4b6-e7d38d3f13e0)

## URL
'https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70'

## Folders - BASH
`mkdir -p {data/{raw/{images,annotations},processed/{tiles,masks},splits,sample_submission},notebooks,models,src/{data,models,scripts,utils,submission},submission}`


## Repository layout
