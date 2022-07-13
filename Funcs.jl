# 微分方程本身所用的数学函数，

function Funcs()

    # f(x) = sin(x) ;
    # f(x)= x*x -1/4
    f(x) = exp(x-0.2)-1
    # f(x,y) = x * sin(x)
    
    # partial of function wrt x
    # df_dx(x) =  2* x #cos(x);
    df_dx(x) = exp(x-0.2)
    # exact solution!!
    yExact(x) = x*x-1/4 # -cos.(x) ;  # 1./(1+exp(-x));

    return f,df_dx,yExact
    
end