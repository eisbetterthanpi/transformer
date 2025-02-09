# transformer
Implementation of the transformer model from the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) in pytorch for sequence to sequence translation.<br /><br />
There are two implementations, one using nn.Transformer and the other from scratch.<br />

### some important implementation details
1) weight initialisation is very imprtant, be sure to init weights after initialising the model parameters
\(nn.init.xavier_uniform_\)<br />
2) set dropout=0 in the scaled dot product attention<br />
3) set bias=False in the Query, Key and Value linear layers<br />

Without these 3, the performance of the transformer will be significantly hindered!<br />

can try using a learnt position embedding instead of the fixed sin/cos position embedding.

<div align="center">
  <div>&nbsp;</div>
  <img src="resources/transformer.png" width="400"/>
  <img src="resources/mha.png" width="200"/>
<!--   <div align="center">original | reconstructed </div> -->
</div>


### extra resources:<br />
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb<br />
https://www.mihaileric.com/posts/transformers-attention-in-disguise/<br />
https://jalammar.github.io/illustrated-transformer/<br />
http://nlp.seas.harvard.edu/2018/04/03/attention.html<br />





