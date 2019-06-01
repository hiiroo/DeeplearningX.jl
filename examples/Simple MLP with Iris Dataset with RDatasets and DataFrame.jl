
using LinearAlgebra
using RDatasets
using Plots
using Deeplearning

iris = dataset("datasets", "iris")

d,dr = @onehot iris[1:end, :Species]

iris_normalized = nothing
for i in 1:4
    iris_normalized = iris_normalized == nothing ? iris[1:end,i]./maximum(iris[1:end,i]) : hcat(iris_normalized, iris[1:end,i]./maximum(iris[1:end,i]))
end

iris_gpu = @cudaarray iris_normalized'

input = zeros(4,1)
dense1 = @dense input 5
dense2 = @dense dense1(input) 3
f(x) = sigmoid(dense2(dense1(x)))

acc_plot = []
loss_plot = []

for e = 1:1000
    misfits=0
    mean_err=0
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
            param .-=0.001*dw
        end
    end
    append!(acc_plot, 1-(misfits/size(iris_gpu)[2]))
    append!(loss_plot, mean_err/size(iris_gpu)[2])
end

plot([loss_plot], linewidth=2, title="Loss")

plot([acc_plot], linewidth=2, title="Accuracy")
