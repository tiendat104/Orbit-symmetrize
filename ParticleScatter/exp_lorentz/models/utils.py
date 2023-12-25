import torch 

def get_inverse_Lorentz(A): # checked
    # A: [B, 4, 4]
    B = A.shape[0]
    
    Q = -torch.eye(4, dtype=torch.float, device=A.device) # [4, 4]
    Q[0,0] = 1.0 
    Q = Q.unsqueeze(0).expand(B, 4, 4) # [B, 4, 4]
    
    A_trans = A.transpose(-1, -2) # [B, 4, 4]
    A_inv = torch.matmul(Q, A_trans)
    A_inv = torch.matmul(A_inv, Q) # [B, 4, 4]
    return A_inv
    
def compute_loss_bound(gs, bound:float): 
    # gs: [B, 4, 4]
    B = gs.shape[0]
    gs = torch.abs(gs) # [B, 4, 4]
    gs = gs - bound 
    gs = torch.nn.ReLU()(gs) # [B, 4, 4]
    
    loss_bound = gs.sum() / B
    return loss_bound