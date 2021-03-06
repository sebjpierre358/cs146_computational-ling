README - zeroshot

HYPERPARAM RATIONALE
40 RNN Size
- I struggled a bit to consistently make 70% accuracy with the same model as
  in multilingual, which had an rnn size of 30. I found increasing this helped,
  and performance was not impacted much.

128 Embedding size
- an increase from the size of 100 in the last model. I believe this increase too
   was helpful without costing much

128 batch size
- this was previously 64. I skimmed through a few papers suggesting that seq2seq
  improved with larger batch sizes. I tried as high as 256 without issue, but results
  seemed worse than with 128 or even 64.

8 Epochs
- another increase from 6 in the multilingual model. I was more consistently able to
  go above 70% accuracy with this many training iterations. There was not issue with
  reaching the 30min training limit here since I removed an extra linear layer after
  the decoder, which seemed not to be helping much.

.003 learning rate
- changed from .002 before. I experimented with a lot of values, but this seemed to
  work better. I do believe accuracy would still be fairly good if changed back to
  .002 based on what I saw.  

Comet.ml urls below

MODEL DISCUSSION
I was able to break past 70% accuracy with both the one-to-one and one-to-many
models, but the one-to-one models were consistently slightly more accurate.
In theory, one-to-many translation should increase the complexity of the
learning task compared to one-to-one since the model is additionally learning
to translate to another language with an almost entirely different set of tokens.
One-to-many translation also entails twice as many training examples compared
to the one-to-one case. Training for the same number of epochs but with more examples
may not be enough to achieve the same level of accuracy as with translating to only
one language.

Best EN -> FR 1-to-1: 73.6% acc, 7.84 perplex
https://www.comet.ml/uk4xfarqr/zeroshot/330051b82d3d425a9f642e12f82567ee

Best EN -> DEU 1-to-1: 72.4% acc, 11.4 perplex
https://www.comet.ml/uk4xfarqr/zeroshot/9a24f393d2dd4e30a83fbadd46ad622e

Best EN -> DEU/FR 1-to-Many: 71.9% acc, 10 perplex
https://www.comet.ml/uk4xfarqr/zeroshot/5637b7ccb2f34b938cd25ff4fc02bf2d

I seemed to be able to replicate zeroshot translation from the Google paper,
but the accuracy seemed much better than I was expecting it to be. I kept the
8 epochs of training I had in the best one-to-one and many-to-one experiments,
switched off to 1 epoch, and never went below 40% accuracy in both cases. I was
a little skeptical that the model could do this well considering the benchmark
is ~10% and the regular non-zeroshot experiments could barely break past 70% in
the best cases. Still, the zeroshot is significantly less accurate than many-to-one
as expected.

Even though examples of Deu -> Frn are never given, the model is capable of making
a shoddy translation since in training it has seen German as a source language
and French as a target, though not as a pair. Perhaps in translation the model
is representing the "meaning" of a sentence similarly despite the linguistic
difference. The embedding matrix being learned gives a relationship between every
word in the bpe corpus, and so even without having seen the explicit language pair,
values in the vector space of embeddings exist such that the model has some of
the knowledge needed to "bridge the gap."  

Zeroshot DEU->FR: 8 epochs 54.8% acc, 33 perplex
https://www.comet.ml/uk4xfarqr/zeroshot/14e6409fb5ed452e984f9009b483a07e

Zeroshot DEU->FR: 3 epochs 40.% acc, 49 perplex
https://www.comet.ml/uk4xfarqr/zeroshot/5b62428552554e739f4887e813314615

Zeroshot DEU->FR: 1 epoch 49.8% acc, 35.9 perplex (results for 1ep varied the most) 
https://www.comet.ml/uk4xfarqr/zeroshot/f639dc09468d4b99a944c5027e0e944e

In the ZeroshotDataset code, it is assumed (from the given corpus files)
that the example order is X -> Y for the source file & Y -> Z from the 2nd file
so that only sentences from the 2nd language will be taken from both to form the
line-for-line correspondence.
