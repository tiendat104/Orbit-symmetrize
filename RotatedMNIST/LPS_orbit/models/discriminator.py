import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator,self).__init__()
    
    def forward(self, M): # checked
        # M: [B, 2, 2]
        # return [B, 4]
        B = M.shape[0]
        
        M11 = M[:,0,0] # [B]
        M12 = M[:,0,1]
        M21 = M[:,1,0]
        M22 = M[:,1,1]
        
        s1 = torch.square(M11) + torch.square(M21) # [B]
        s2 = torch.square(M12) + torch.square(M22) # [B]
        s3 = M11 * M12 + M21 * M22 # [B]
        s4 = M11 * M22 - M12 * M21 # [B]
        
        out = torch.cat((s1[:,None], s2[:,None], s3[:,None], s4[:,None]), dim=1) # [B, 4]
        return out
