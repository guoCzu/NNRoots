include("Forward.jl")
include("Sig.jl")
# include("Dsig.jl")
function backPropagate(wH, bH, wO,n,x,f,df_dx,eta,droprate,epoch)
    
    
    # # grad of output layer weights
    nO = length(wO) # 隐藏层的神经元个数
    dwO = zeros(nO);
    err_wO = zeros(nO); 
    # # grad of hidden layer weights
    sz_In_H = size(wH)
    nIn,nH = sz_In_H[2],sz_In_H[1]
    dwH = zeros(nH,nIn);
    err_wH = zeros(nH,nIn);
    dbH = zeros(nH)
    err_bH = zeros(nH)
    ###
   

    # drop learning rate by half every 20 epochs
    # eta = eta*(1/2)^floor(epoch/droprate);
    # eta = eta * 1.0/2^(floor((epoch+1)/5000))

    # loop over 
    for i = 1:n # x 的100个点。
        # feedforward 
        # current
        # while true
            aH,zH,aOut,zO = feedForward(wH, bH,wO,x[i],f(x[i]),df_dx(x[i]));
            # aO = a_out 
            # 先把三个输入作为一个向量
            xInV = [x[i],f(x[i]),df_dx(x[i])]
            ###
            # gradients of network parameters
            err_wO = aH 
            for jn=1:nH
                for in=1:nIn
                    # err_wH[jn,in] = wO[jn]*dsig(zH[jn])*xInV[nIn]
                    err_wH[jn,in] = wO[jn]*dRelu(zH[jn])*xInV[nIn]
                end 
            end
            # err_bH = wO .* (dsig.(zH))
            err_bH = wO .* (dRelu.(zH))
        
            # update llhlearning rate
            
            # gradient descent
            grad = aOut-xInV[1]+xInV[2]/xInV[3]
            # output layer weights
            dwO = grad * err_wO
            wO = wO - eta*dwO;
            # hidden layer weights
            dwH = grad * err_wH
            wH = wH - eta*dwH;
            # hidden layer bias
            dbH = grad * err_bH
            bH = bH - eta*dbH;
            ###
            # 更新 x[i]的值
            # if abs(x[i]-aOut)> 1.0e-2
            #     x[i] = x[i] - xInV[1]+xInV[2]/xInV[3] 
            # else 
            #     break
            # end
        # end
        
    end
    # wHs = [wH1,wH2,wH3 ]
    # bHs = [bH1,bH2,bH3 ]
    return wH,bH,wO
end