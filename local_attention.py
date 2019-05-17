import torch
import torch.nn as nn


class Local_attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions):
        super(Local_attention, self).__init__()

        self.linear_in = nn.Linear(dimensions, dimensions * 2, bias=False)
        self.att_vec = torch.rand(dimensions * 2, 1)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def masked_softmax(self, T):
        T[T==0] = -10^20
        return self.softmax(T)
        
    def forward(self, context):
        """
        Args:
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, pron_len, dimensions = context.size()


        context = context.view(batch_size * pron_len, dimensions)
        context = self.linear_in(context)
        context = context.view(batch_size, pron_len, 2 * dimensions)

        attention_scores = context.matmul(self.att_vec)        
        attention_weights = self.masked_softmax(attention_scores)

        #print(attention_weights[:3])
        mix = torch.bmm(attention_weights.transpose(1,2).contiguous(), context).view(batch_size, 2 * dimensions)
        attention_weights = attention_weights.view(batch_size, pron_len)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(mix)
        output = self.tanh(output)

        return output, attention_weights
