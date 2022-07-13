include("Forward.jl")
include("Backpropagate.jl")
include("MyConstants.jl")
include("Funcs.jl")
using Plots
using Distributions
using Dates
# Equation Parameters
# y' = f(y,x)
# -----------
# by guoxiaobo, etc.  @czu, 2022.5.15
# -----------
# clear everything
function Train(x,f,df_dx,wH, bH, wO, a_out)
    
    nBP = MyConst.nBP

    # Number of Training Points
    N = MyConst.Nx;

    # Network Parameters
    # intial learning rate
    eta = MyConst.eta
    etaX = MyConst.etaX
    # drop rate
    droprate = MyConst.droprate
    # # hidden layer 神经数。

    # feedforward over batches
    while true
        # for i = 1:N
        #     temp1,temp2,a_out[i],temp3 = feedForward(wH, bH,wO,x[i],f(x[i]),df_dx(x[i]));
        # end

        ############################## 
        # # backpropagation algorithm
        for i = 1:nBP
            # wHOld,bHOld,wOOld = wH,bH,wO
            wH,bH,wO = backPropagate(wH, bH, wO,N,x,f,df_dx,eta,droprate,i);
            # max1 = maximum(abs.(wH-wHOld)./abs.(wH))
            # max2 = maximum(abs.(bH-bHOld)./abs.(bH))
            # max3 = maximum(abs.(wO-wOOld)./abs.(wO))
            # if max1 <1.0e-3 && max2<1.0e-3 && max3<1.0e-3 
            #     # println("最大训练次数为： ",i)
            #     # break 
            # end
        end
        # feedforward over training inputs
        for i = 1:N
            aH,zH,a_out[i],z_out = feedForward(wH, bH,wO,x[i],f(x[i]),df_dx(x[i]));
        end
        if abs(x[1]-a_out[1])> 1.0e-2
            x[1] = x[1] - etaX * (x[1]-f(x[1])/df_dx(x[1]) )
        else 
            break
        end
    end

    return a_out
end

#     # Plot Actual vs. ANN Solution       
#     p0 = plot(1)
#     p2 = plot!(p0,x,y(x),color="red",label = "analysis solution",legend=true)
#     p2 = plot!(p2,x,C_1 .+ x.*a_out,color="black",label = "neural solution",legend=true)
#     # savefig("curve.png")
#     xlabel!("x")
#     ylabel!("y")
#     display(p2)
#     savefig("p2.png")
#     title!("Exact vs. ANN-computed solution to y' = y")
#     # exit()
#     # legend!("Exact","ANN","location","northwest")
# # ######################################
# #     # Error Plot
#     n_err = N;
#     # sample
#     x_err = Array{Float64}(undef,N)
#     x_err .= x
#     # x_err = 0:1/n_err:1 -1/n_err
#     # x_err = linspace(0,1,n_err)";
#     a_out_err = zeros(n_err,1);
#     # feedforward over error-evaluating inputs
#     for i = 1:n_err
#         a_H,z_H,a_out_err[i],z_out = feedForward(w_H,b_H,w_out,x_err[i]);
#     end
#     # get errors
#     err = abs.(y(x_err) .- (C_1 .+ x_err.*a_out_err));

#     p2= plot!(p2,x_err,err,color=:green)
#     xlabel!("x")
#     ylabel!("error")

#     title!("Absolute Error of ANN-computed solution to y"*" = y")
##################################################################
# #     # Extrapolation Plot
#     m = 2* N;
#     # ex = Array{Float64}(undef,N)
#     # ex .= x
#     # ex = linspace(0,10,N)';
#     ex = range(1, 10, length=m)
#     a_out = zeros(m,1);
#     # feedforward over extrapolation points
#     for i = 1:m
#         a_H,z_H,a_out[i],z_out = feedForward(w_H,b_H,w_out,ex[i]);
#     end

#     # plot!(ex,y(ex))
#     p2 = plot!(p2,ex,IC .+ ex.*a_out)
#     xlabel!("x")
#     ylabel!("y")

#     title!("Extrapolation of ANN-computed solution to y' = y")
#     # legend("Exact","ANN","location","northwest")
# ######################################
    # return w_H
# end


##############################

# println("开始运行。 ",now())
# main()
# println("结束运行。 ",now())