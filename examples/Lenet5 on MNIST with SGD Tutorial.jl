
using AutoGrad
using Deeplearning
using Plots
import CSV

train = CSV.read("datasets/digit-recognizer/train.csv")
# test = CSV.read("datasets/digit-recognizer/test.csv")

d,dr = @onehot train.label

train_normalized = nothing
for i in 1:100
    normalized = reshape(train[i, 2:end]./(maximum(train[i,2:end])+0.1), (28,28,1,1))
    if train_normalized == nothing
        train_normalized = normalized
    else
        train_normalized = cat(train_normalized, normalized, dims=4)
    end
end

size(train_normalized)

train_gpu = @cudaarray train_normalized

input = @cudaarray rand(28, 28, 1)
w1 = ((rand(3,3,6).-0.5)./100)
w2 = ((rand(3,3,16).-0.5)./100)

conv1 = @convolve input w1 (2,2) (1,1)
# pool1 = @maxpool conv1(input) (2,2) (1,1) #Replaced poolings with strided convolutions
conv2 = @convolve conv1(input) w2 (2,2) (1,1)
# pool2 = @maxpool conv2(pool1(conv1(input))) (2,2) (1,1) #Replaced poolings with strided convolutions
dense1 = @dense conv2(conv1(input)) 120
dense2 = @dense dense1(conv2(conv1(input))) 84
dense3 = @dense dense2(dense1(conv2(conv1(input)))) 10
layers=[dense3,dense2,dense1,conv2,conv1]

f(x) = softmax(dense3(relu(dense2(relu(dense1((relu(conv2((relu(conv1(x))))))))))))

acc_plot = []
loss_plot = []

dff(x,y) = @diff squared_diff(f(x), y)

for e = 1:100
    println("Epoch ", e)
    misfits=0
    mean_err=0
    for i=1:size(train_gpu)[4]
        o = f(train_gpu[:,:,:,i])
        olabel = @onehotdecode dr o
        
        if(olabel == nothing || olabel!=train[i,:label])
           misfits+=1 
        end
        
        err = squared_diff(o, d[train[i,:label]])
        mean_err+=err
        
        dv = dff(train_gpu[:,:,:,i], d[train[i,:label]])
        dparams = @parameters dv

        for pidx in 1:length(dparams[1:end-1])
            dw = grad(dv, dparams[pidx])
            layers[pidx].matrix .-=0.1*dw
        end
    end
    append!(acc_plot, (1-(misfits/size(train_gpu)[4]))*100)
    append!(loss_plot, mean_err/size(train_gpu)[4])
end
dff(x,y) = @diff squared_diff(f(x), y)
batch_size = 25

for e = 1:10
    println("Epoch ", e)
    misfits=0
    mean_err=0
    for i=1:batch_size:size(train_gpu)[4]
        b = train_gpu[:,:,:,i:(i+batch_size-1)]
        o = mapslices(x->f(x),train_gpu[:,:,:,i:(i+batch_size-1)],dims=[1,2,3])

        olabel = mapslices(x->decodeonehot(dr, x), o, dims=[1,2,3])[:]

        misfits+=count((olabel .== nothing) .+ (olabel.!=train[1:(1+batch_size-1),:label]))

        ytrues = hcat(map(x->d[x], train[1:batch_size,:label])...)
        err = sum([squared_diff(o[:,1,1,ys], ytrues[:,ys]) for ys =1:batch_size])/batch_size
        mean_err+=err
                
        dfs = [dff(train_gpu[:,:,:,eb], ytrues[:,eb]) for eb=1:batch_size]
        dparams = [@parameters dfs[eb] for eb=1:batch_size]

        dgrads = [[grad(dfs[eb],dparams[eb][ep]) for ep=1:length(dparams[eb])] for eb=1:batch_size]

        catted = map(+,dgrads...)
        
        for pidx in 1:length(layers)
            layers[pidx].matrix .-=0.1*catted[pidx]
        end
    end
    append!(acc_plot, (1-(misfits/size(train_gpu)[4]))*100)
    append!(loss_plot, mean_err/size(train_gpu)[4])
end
println(acc_plot[end])
println(loss_plot[end])

plot([acc_plot], linewidth=2, title="Accuracy")

plot([loss_plot], linewidth=2, title="Loss")
