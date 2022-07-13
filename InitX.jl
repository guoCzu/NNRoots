

function InitX()
    # Training Points
    Nx = MyConst.Nx 

    x= Array{Float64}(undef,Nx)
    dx = 1.0/Nx
    for j = 1:Nx
        x[j] = (j-1) * dx #+ dx # 这里加 dx 为了避免从0点开始
    end
    # x = 0:dx:6*π
    x[1] = 2.7
    return x
    
end
########################
function InitValues() # 微分方程本身的初始条件
    # Initial Condition
    # IC = -1.0;
    C_1 = -1.0 

    return C_1
    
end