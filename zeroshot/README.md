README - zeroshot

I was able to break past 70% accuracy with both the one-to-one and one-to-many
models, but the one-to-one models were consistently slightly more accurate.
In theory, one-to-many translation should increase the complexity of the
learning task compared to one-to-one since the model is additionally learning
to translate to another language with an almost entirely different set of tokens.
One-to-many translation also entails twice as many training examples compared
to the one-to-one case. Training for the same number of epochs but with more examples
may not be enough to achieve the same level of accuracy as with translating to only
one language.

Best EN -> FR 1-to-1
https://www.comet.ml/uk4xfarqr/zeroshot/330051b82d3d425a9f642e12f82567ee

Best EN -> DEU 1-to-1
https://www.comet.ml/uk4xfarqr/zeroshot/9a24f393d2dd4e30a83fbadd46ad622e

Best EN -> DEU/FR 1-to-Many
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

Zeroshot - 8 epochs
https://www.comet.ml/uk4xfarqr/zeroshot/14e6409fb5ed452e984f9009b483a07e

Zeroshot - 3 epochs
https://www.comet.ml/uk4xfarqr/zeroshot/5b62428552554e739f4887e813314615

Zeroshot - 1 epoch
https://www.comet.ml/uk4xfarqr/zeroshot/a76d20408f6a40279ed4ac93a62c0fb6

In the ZeroshotDataset code, it is assumed (from the given corpus files)
that the example order is X -> Y for the source file & Y -> Z from the 2nd file
so that only sentences from the 2nd language will be taken from both to form the
line-for-line correspondence. 
