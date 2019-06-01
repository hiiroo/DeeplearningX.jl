
using LinearAlgebra
using RDatasets
using Plots
using AutoGrad
using Deeplearning

iris = dataset("datasets", "iris")

d,dr = @onehot iris[1:end, :Species]

iris_normalized = nothing
for i in 1:4
    iris_normalized = iris_normalized == nothing ? iris[1:end,i]./maximum(iris[1:end,i]) : hcat(iris_normalized, iris[1:end,i]./maximum(iris[1:end,i]))
end

iris_normalized

iris_gpu = @cudaarray iris_normalized'

input = zeros(4,1)
dense1 = @dense input 5
dense2 = @dense dense1(input) 3
f(x) = sigmoid(dense2(dense1(x)))

acc_plot = []
loss_plot = []

lrate = 0.1#10e-8
alpha = 0.001
beta1 = 0.5#0.9
beta2 = 0.5#0.999
mt = 0
vt = 0
t = 0
for e = 1:100
    misfits=0
    mean_err=0
    t+=1
    for i=1:1:size(iris_gpu)[2]
        o = f(iris_gpu[1:end,i])
        olabel = @onehotdecode dr o
        
        if(olabel!=iris[i,:Species])
           misfits+=1 
        end
        
        err = squared_diff(o, d[iris[i,:Species]])
        mean_err+=err
        
        df = @diff squared_diff(f(iris_gpu[1:end,i]), d[iris[i,:Species]])
        params = @parameters df
        
        for param in params[1:end-1]
            dw = grad(df, param)
            mt = beta1*mt.+(1-beta1).*dw
            vt = beta2*vt.+(1-beta2).*(dw.^2)
            mt = mt./(1-beta1^t)
            vt = vt./(1-beta2^t)
            param .-=alpha.*(mt./(sqrt.(vt).+lrate))
        end
    end
    append!(acc_plot, (1-(misfits/size(iris_gpu)[2]))*100)
    append!(loss_plot, mean_err/size(iris_gpu)[2])
end

plot([loss_plot], linewidth=2, title="Loss")

plot([acc_plot], linewidth=2, title="Accuracy")

acc_plot[end]
