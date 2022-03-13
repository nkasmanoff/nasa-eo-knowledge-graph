import torch
import torch.nn as nn


class transE(nn.Module):

    def __init__(self, entity_size, relation_size, embedding_size):
        super(transE, self).__init__()

        self.W_en = nn.Embedding(entity_size, embedding_size)
        self.W_re = nn.Embedding(relation_size, embedding_size)

    def forward(self, pos_h, pos_r,  pos_t, neg_h, neg_r, neg_t): 
        
        pos_h_e = self.W_en(pos_h)
        pos_t_e = self.W_en(pos_t)
        pos_r_e = self.W_re(pos_r)
        neg_h_e = self.W_en(neg_h)
        neg_t_e = self.W_en(neg_t)
        neg_r_e = self.W_re(neg_r)
            
        # TODO - can make the weights above encode non-linearity, but for now it is just the dot product. 
        posError = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2)
        negError = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2) # this value should be maximized
        return posError # tutorial only uses posError, keeping it for consistency.  