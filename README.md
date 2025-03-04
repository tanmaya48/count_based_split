# count_based_split

This Repo is to implement splitting of Object detection dataset which takes into account the distribution of objects as well.

### Evaluation:

This will be evaluated on the COCO dataset, by comparing performance of a model trained on 2% subset of train set and evaluated on the val set.

For the control, Random split will be used to compare the results.
This will be run multiple times with different seeds to get fair results.
