import torch
import torch.nn as nn  # pylint: disable=R0402:consider-using-from-import
import torch.nn.functional as F

def generating1(M):
    # M: [B, 4, 4]
    B = M.shape[0]
    eng = torch.diag(torch.tensor([1., -1., -1., -1.])).to(M.device) # [4, 4]
    eng = eng.unsqueeze(0).expand(B, 4, 4) # [B, 4, 4]

    A = torch.matmul(eng, M) # [B, 4, 4]
    A = torch.matmul(M.transpose(-1, -2), A) # [B, 4, 4]
    A = A.flatten(start_dim=1) # [B, 16]
    return A

def get_upper_index(M): 
    # M: [B, n, n]
    B = M.shape[0]
    n = M.shape[1] 
    
    idx0 = torch.arange(n*n, device=M.device).reshape(n, n) # [n, n]
    # get mask
    idx1 = torch.arange(1, n+1, device=M.device)[None,:].expand(n, n) # [n, n]
    idx2 = torch.arange(n, device=M.device)[:,None] # [n, 1]
    mask_upper = idx1 > idx2 # [n, n]
    
    upper_index = idx0[mask_upper] # [n*(n+1)/2]
    return upper_index

def get_order1(M): 
    # M: [B, 4, 4]
    B = M.shape[0]
    A = generating1(M) # [B, 16]
    upper_idx4 = torch.tensor([0, 1, 2, 3, 5, 6, 7, 10, 11, 15], device=M.device)
    A1 = torch.index_select(A, 1, upper_idx4) # [B, 10]
    return A1 # [B, 10]

def get_order2(A1):
    # A1: [B, 10]
    A2 = torch.matmul(A1[:,:,None], A1[:,None,:]) # [B, 10, 10]
    upper_idx10 = get_upper_index(A2) # [55] 
    A2 = A2.flatten(start_dim=1) # [B, 100]
    A2 = torch.index_select(A2, 1, upper_idx10) # [B, 55]
    return A2 # [B, 55]

def get_order4(A2): 
    # A2: [B, 55]
    A4 = torch.matmul(A2[:,:,None], A2[:,None,:]) # [B, 55, 55]
    upper_idx55 = get_upper_index(A4) # [1540]
    A4 = A4.flatten(start_dim=1) # [B, 3025]
    A4 = torch.index_select(A4, 1, upper_idx55) # [B, 1540]
    return A4 # [B, 1540]

def generating2(M): 
    # M: [B, 4, 4]
    B = M.shape[0]
    A1 = get_order1(M) # [B, 10]
    
    # order 2 tensor 
    A2 = get_order2(A1) # [B, 55]
    
    out = torch.zeros(B, 65, device=M.device) # [B, 65]
    out[:,:10] = A1
    out[:,10:] = A2
    return out # [B, 65]

def generating4(M): 
    # M: [B, 4, 4]
    B = M.shape[0]
    A1 = get_order1(M) # [B, 10]
    # order 2 tensor 
    A2 = get_order2(A1) # [B, 55]
    # order 4 tensor 
    A4 = get_order4(A2) # [B, 1540]

    n1 = A1.size(1)
    n2 = A2.size(1)
    n4 = A4.size(1)
    out = torch.zeros(B, n1+n2+n4, device=M.device) # [M, 1605]
    out[:,:n1] = A1 
    out[:,n1:n1+n2] = A2 
    out[:,n1+n2:] = A4 
    return out # [B, 1605]

def generating5(M): 
    # M: [B, 4, 4]
    B = M.shape[0]
    A1 = get_order1(M) # [B, 10]
    # order 2 tensor 
    A2 = get_order2(A1) # [B, 55]
    # get order 4 diagonal
    C = torch.index_select(A1, 1, torch.tensor([0, 4, 7, 9], device=M.device)) # [B, 4]
    A3 = torch.pow(C, 3) # [B, 4]
    A4 = torch.pow(C, 4) # [B, 4]
    
    n1 = A1.size(1)
    n2 = A2.size(1)
    out = torch.zeros(B, n1+n2+8, device=M.device) # [M, 73]
    out[:,:n1] = A1 
    out[:,n1:n1+n2] = A2 
    out[:,n1+n2:n1+n2+4] = A3
    out[:,n1+n2+4:] = A4 
    return out # [M, 73]

def compute_generating(M, type=1):
    if type==1:
        return generating1(M) # [B, 16]
    elif type == 2: 
        return generating2(M) # [B, 65]
    elif type == 4:
        return generating4(M) # [B, 1605]
    elif type == 5:
        return generating5(M) # [B, 73]

if __name__ == "__main__":
    device = torch.device("cuda:0")
    M = torch.randn(5, 4, 4, device=device)
    A = generating4(M)
    print("A.shape = ", A.shape)