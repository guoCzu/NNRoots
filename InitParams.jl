include("MyConstants.jl")
function InitParams(nIn,nH,nOut)
    Nx = MyConst.Nx # x 的点数

    # H_len = length(C_H_idx)

    # wIn = zeros(C_in_idx)

    wH = zeros(nH,nIn)
    bH = zeros(nH)
    # wH2 = zeros(C_H_idx[2])
    # wH3 = zeros(C_H_idx[3]) # 隐藏层


    wO = zeros(nH)    # 输出层
    # H1= C_H_idx[1]
    # H2= C_H_idx[2]
    # H3= C_H_idx[3]
    # for i =1:H_len
    bH = randn(nH,1)
    # w_H = normrnd(0,1/sqrt(nH),[nH,nIn]);
    norm1 = Normal(0,1/sqrt(nH)) # μ = 0.0 σ=1.0
    wH = rand(norm1,(nH,nIn)) #(1,H)'
        ###
        # bH2 = randn((H2,1))
        # n2 = Normal(0,1/sqrt(H1)) # μ = 0.0 σ=1.0
        # wH2 = rand(n2,(H2,H1)) #(1,H)'
        # ###
        # bH3 = randn(H3,1)
        # n3 = Normal(0,1/sqrt(H2)) # μ = 0.0 σ=1.0
        # wH3 = rand(n3,(H3,H2)) #(1,H)'

    # end

    wO = randn(nH,1) # H1 代表一个隐藏层
    
    aO = zeros(Nx,1);    #  a_out

    # wHs = wH #[wH1,wH2,wH3]     # 总的隐藏层
    # bHs = bH # [bH1,bH2,bH3]

    return  wH, bH, wO, aO
end