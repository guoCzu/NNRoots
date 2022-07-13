function sig(z)

    a = 1.0 /(1.0 + exp(-z));

    return a
end
############################
function dsig(z)
    a = sig(z) * (1.0 -sig(z));
    return a 
end
#######################
function d2sig(z)
    a = sig(z) * (1.0 -sig(z)) * (1.0 - 2.0* sig(z) );
    return a 
end
#########################
function Relu(z)
    # if z<=0.0
    #     return 0.0
    # else 
    #     return z 
    # end
    ### 上面是 经典 Relu ，下面是 Leaky Relu
    if z<0.0 
        ε = 0.01
        return 0.0 #ε * z
    else 
        return z
    end 
    # ε = 0.1 
    # zNew = ε * z
    # a = maximum([zNew,z])
    # return a 
end
#########################
function dRelu(z)
    # if z<=0.0 
    #     return 0.0
    # else 
    #     return 1.0 
    # end 
    ### 上面是 经典 Relu ，下面是 Leaky Relu
    if z<0.0 
        ε = 0.01
        return 0.0 # ε 
    else 
        return 1.0
    end 
end
# def relu_prime(data, epsilon=0.1):

# if 1. * np.all(epsilon < data):

# return 1

# return epsilon