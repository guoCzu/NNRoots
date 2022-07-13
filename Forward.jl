include("Sig.jl")
function feedForward(wH, bH,wO,x_n,f_xn,df_dxn)
    # nHidLayers = length(bHs) # 隐藏层个数
    # zHs = Array{typeof(bHs[1])}(undef,(nHidLayers,1))
    # aHs = Array{typeof(bHs[1])}(undef,(nHidLayers,1))
    # for i =1:nHidLayers
    #     zHs[i] = zeros(length(bHs[i]),1)
    #     aHs[i] = zeros(length(bHs[i]),1)
    # end
    xIn = [x_n,f_xn,df_dxn]
    # 从输入到隐藏层传输
    zH = wH*xIn .+ bH
    # aH = sig.(zH)
    aH = Relu.(zH)

    
    # for i =2 :nHidLayers
    #     zHs[i] = wHs[i] * aHs[i-1] + bHs[i]
    #     aHs[i] = sig.(zHs[i])
    # end
    ###
    z_out = wO' * aH ;
    a_out = z_out[1]
    # # weighted inputs to hidden layer

    
    return aH,zH,a_out,z_out
end