# APET
Atomic Positional Embedding-based Transformer
Introduced by Cui Yaning. The paper is under review.

## V1 No change
## V2 and after, train+valid instead of train+test
newposition1: xsin xcos ysin ycos zsin zcos-> xsin1 xcos1 ysin1 ycos1 zsin1 zcos1 xsin2...
newposition2: tensor+posx+posy+posz
newposition3: based on 2,inv_freq.length*3 -> x+y+z,diff freq reps diff posk
## V3 +atom feature in the initail input src
## V4 +atom feature in every encoder layer