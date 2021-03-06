multilingual

Hyperparam Rationale:
- The Adam params you might see in the Comet experiments are carryovers from
  my old Seq2Seq implementation in tensorflow. I originally started this assignment
  in pytorch, but inexplicably faced low acc/high perplexity on my model and wanted
  to use a tensorflow implementation to compare for issues. I found that choosing not to do
  pack/pad sequence in the model completely fixed this, though I was not able to
  determine why this was an issue. So please ignore the extra Adam params.

- The learning rate I settled on was .002, and I fluctuated a bit between
  .006 - 0.001 previously. I found .001 to be a little to restrictive with learning,
  and values above .003 seemed to yield accuracy similar to with .001.

- Embedding size 100 was maintained from the start, and I found no need to change
  it. When considering how to bump accuracy past the 70% threshold I reasoned that
  lowering the embed size would hurt performance given less features for each word
  to put thru the network. Might also be additionally bad given the larger than
  normal vocab size of a joint bpe. I avoided increasing the embed size to keep
  training time not too long.

- For the vanilla and eng-frn bpe datasets, I had the model run for 6 epochs,
  which trained in about 15mins. Fewer epochs were not able to get the model
  reliably past 70% accuracy, and I avoided values past 6 for lower training time
  and to avoid possible overfit. The best multilingual training took ~20mins for
  3 epochs, but was able to reach past 70%. Values past this would quickly have
  neared the 30min limit, and likely would not have been necessary for reaching the
  accuracy threshold

- The rnn_size of 30 was a somewhat arbitrary pick after looking through the raw
  bpe data files. I reasoned it should be a bit longer than in parse_reranker since
  the <w/> tags and sub-word tokens would extend sequences quite a bit. Training
  with the vanilla set revealed that all sequences in that set were actually <30.

VANILLA DATASET:
https://www.comet.ml/uk4xfarqr/multilingual/2bb3ff68eb034cf99e9b31386bff1667
ENG-FR BPE:
https://www.comet.ml/uk4xfarqr/multilingual/ae7fe6af570d4e52b3935369b90d18c5
MULTILINGUAL BPE:
https://www.comet.ml/uk4xfarqr/multilingual/ea34422d1d9d42e68317729282f92447

BPE vs Traditional
The translating on the byte-pair encoded vocab certainly boosts accuracy significantly
compared to the traditional two-pair model. Additionally, being able to translate over
one joint vocab allows for easier setup of multilingual translation, a more cumbersome
task with split vocabularies. An interesting consequence of sharing a large vocab for
bpe datasets is the case of shared tokens between languages. For example, "a" in
English (the article) is different from "a" in French (actually a verb), and so the model
needs to learn how to distinguish between the two in the bpe case, whereas this is added
learning complexity is avoided in datasets with vocabs of unique languages. This
problem is alleviated in part with target sentence tags though, and it should be
noted that the "same token" problem already exists within a language for homographs,
which attention and the concept of querying keys for the correct "value" of a word
given a context help distinguish. Fundamentally, translating on a BPE is great for
avoiding UNKing words, which is desirable for some types of articles, names, and
online/social media text where specific, new, or specialized words abound.
