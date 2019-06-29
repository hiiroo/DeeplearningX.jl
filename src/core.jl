#=
MIT License

Copyright (c) 2019 Ali Mert Ceylan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

"
"
function LogicalIndices(boolarr)
    arr_size = size(boolarr)
    c = reshape(collect(1:prod(size(boolarr))), (prod(size(boolarr)),1))
    return c[reshape(boolarr, (prod(size(boolarr)),1))]
end


_kernelsize(kernelsize::Tuple{Int, Int}, dilations::Tuple{Int, Int}) = ((dilations .* kernelsize) .- (dilations.-1))
# _featuremapsize(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}, strides::Tuple{Int, Int}) = div.(inputsize .- kernelsize, strides) .+ 1
_featuremapsize(inputsize, kernelsize, strides) = map(x->x[1] != nothing ? div(x[1] - x[2], x[3]) + 1 : nothing, zip(inputsize, kernelsize, strides))


"Creates convolution matrix according to given input, kernel, stride and dilation configuration"
function im2col!(conv_matrix, conv_kernel, input_size, kernel_size, strides, dilations, fm_size)
    margin = input_size.%fm_size
    conv_matrix.=0
    t1 = ongpu(mzerosf32(input_size))
    for ci in 1:size(conv_kernel)[4]
        for ki in 1:size(conv_kernel)[3]
            idx = 1
            for i in 1:strides[1]:(fm_size[1]*strides[1])
                for j in 1:strides[2]:(fm_size[2]*strides[2])
                    t1.=0
                    t1[i:dilations[1]:(i+kernel_size[1]-1), j:dilations[2]:(j+kernel_size[2]-1)] = conv_kernel[:,:,ki,ci]'
                    conv_matrix[idx,:,ki,ci] .= t1'[:]
                    idx+=1
                end
            end
        end
    end
end


"Fetches convolution kernel from given convolution matrix, input_size, kernel size, stride and dilation configuration"
function col2im!(conv_matrix, conv_kernel, input_size, weight_size, strides, dilations, fm_size)
    margin = input_size.%fm_size
    kernel_size = size(conv_kernel)

    conv_kernel .= 0
    for ci in 1:kernel_size[4]
        for ki in 1:kernel_size[3]
            cki = 1
            for i in 1:strides[1]:(fm_size[1]*strides[1])
                for j in 1:strides[2]:(fm_size[2]*strides[2])
                    t1 = reshape(conv_matrix[cki,:,ki,ci], input_size[end:-1:1])
                    conv_kernel[:,:,ki,ci].+=t1'[i:dilations[1]:(i+kernel_size[1]-1), j:dilations[2]:(j+kernel_size[2]-1)]
                    cki+=1
                end
            end
            conv_kernel[:,:,ki,ci]./=(size(conv_matrix)[1])
        end
    end
end


function conv(w, x;s=(1,1),d=(1,1))
    x_size = size(x)
    x_channels = x_size[3]
    x_batch = x_size[4]

    kernel_size = _kernelsize(size(w)[1:2], d)
    fm_size = _featuremapsize(x_size[1:2], kernel_size, s)

    cm_size = (prod(fm_size), prod(x_size[1:2]), size(w)[3:end]...)
    cm_channels = cm_size[4]    
    conv_mat=ongpu(mzerosf32(cm_size...))
    
    im2col!(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)

    reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))
    reshaped_conv_mat = reshape(conv_mat, (cm_size[1], prod(cm_size[2:3]), cm_size[4]))
    return reshape(cat([reshape(gemm('N', 'N', Float32(1), reshaped_conv_mat[:,:,cmi], reshaped_x), (prod(fm_size), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (fm_size..., cm_channels, x_batch))
end


function convx(w, x, dy;s=(1,1),d=(1,1))
    x_size = size(x)
    x_channels = x_size[3]
    x_batch = x_size[4]

    kernel_size = _kernelsize(size(w)[1:2], d)
    fm_size = _featuremapsize(x_size[1:2], kernel_size, s)

    cm_size = (prod(fm_size), prod(x_size[1:2]), size(w)[3:end]...)
    cm_channels = cm_size[4]    
    conv_mat=ongpu(mzerosf32(cm_size...))
    
    im2col!(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)
    conv_mat = permutedims(conv_mat, (2,1,3,4))
    conv_mat = permutedims(conv_mat, (1,2,4,3))
    cm_size = size(conv_mat)
    cm_channels = cm_size[4]    
    reshaped_conv_mat = reshape(conv_mat, (cm_size[1], prod(cm_size[2:3]), cm_channels))

    dy_size = size(dy)
    dy_batch=dy_size[4]
    reshaped_dy = reshape(dy, (prod(dy_size[1:3]), dy_batch))
    
    return reshape(cat([reshape(gemm('N', 'N', Float32(1), reshaped_conv_mat[:,:,cmi], reshaped_dy), (prod(x_size[1:2]), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (x_size[1:2]..., cm_channels, x_batch))
end


function convw(w, x, dy;s=(1,1),d=(1,1))
    x_size = size(x)
    x_channels = x_size[3]
    x_batch = x_size[4]

    kernel_size = _kernelsize(size(w)[1:2], d)
    fm_size = _featuremapsize(x_size[1:2], kernel_size, s)

    cm_size = (prod(fm_size), prod(x_size[1:2]), size(w)[3:end]...)
    cm_channels = cm_size[4]    
    conv_mat=ongpu(mzerosf32(cm_size...))
    
    dy = permutedims(dy, (1,2,4,3))
    dy_size = size(dy)
    dy_batch=dy_size[3]
    dy_channels=dy_size[4]
    reshaped_dy = reshape(dy, (prod(size(dy)[1:2]), dy_batch, dy_channels))
    
    reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))'
    im2col!(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)
    
    dw = cat([reshape(gemm('N', 'N', Float32(1), reshaped_dy[:,:,dmi], reshaped_x), (prod(dy_size[1:2]), prod(x_size[1:2]), x_channels, 1)) for dmi in 1:dy_channels]...,dims=4)
    col2im!(dw, w, x_size[1:2], size(w)[1:2], s, d, fm_size)
    return w
end


@primitive conv(w,x;args...),dy convw(w,x,dy;args...) convx(w,x,dy;args...) # maxpoolx(input, window, strides, dilations, y, dy)
@zerograd convx(w,x,dy;args...)
@zerograd convw(w,x,dy;args...)


#Pooling operations

function findmaxidxs(x, xis)
    xis[findmax(x)[2]]
end

function findmeanidxs(x, xis)
    xis
end

function poolidxs(op::Function, input, window::Tuple{Int, Int}, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int}, input_size::Tuple{Int, Int}, kernel_size::Tuple{Int, Int}, margin::Tuple{Int, Int})
    api = CartesianIndices(input_size)
    ap = []
    @inbounds for i in 1:strides[2]:(input_size[2]-margin[2])
        @inbounds for j in 1:strides[1]:(input_size[1]-margin[1])
            @views t1 = input[j:dilations[2]:(j+kernel_size[2]-1), i:dilations[1]:(i+kernel_size[1]-1)]
            @views t1idxs = api[j:dilations[2]:(j+kernel_size[2]-1), i:dilations[1]:(i+kernel_size[1]-1)]
            @views opidxs = op(t1, t1idxs)
            push!(ap, opidxs)
        end
    end

    return ap
end
@zerograd poolidxs(op::Function, input, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int})

function maxpool(input, window, strides, dilations)
    input_size = size(input)
    kernel_size = _kernelsize(window, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size
    
    y = reshape(map(opidx->maximum(input[opidx]), poolidxs(findmaxidxs, input, window, strides, dilations, input_size, kernel_size, margin)), (fm_size...))
    return ongpu(y)
end

function maxpoolx(input, window, strides, dilations, dy, y)
    input_size = size(input)
    kernel_size = _kernelsize(strides, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    opidxs = poolidxs(findmaxidxs, input, window, strides, dilations, input_size, kernel_size, margin)
    dx = mzeros(input_size)
    map(opidxp->dx[opidxp[1]]=opidxp[2], zip(opidxs,y))
    return ongpu(dx)
end

@primitive maxpool(input, window, strides, dilations),dy,y maxpoolx(input, window, strides, dilations, y, dy)
@zerograd maxpoolx(input, window, strides, dilations, dy, y)

function avgpool(input, window, strides, dilations)
    input_size = size(input)
    kernel_size = _kernelsize(window, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    y = reshape(map(opidx->mean(input[opidx]), poolidxs(findmeanidxs, input, window, strides, dilations, input_size, kernel_size, margin)), (fm_size...))
    return ongpu(y)
end

function avgpoolx(input, window, strides, dilations, dy, y)
    input_size = size(input)
    kernel_size = _kernelsize(window, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    opidxs = poolidxs(findmeanidxs, input, window, strides, dilations, input_size, kernel_size, margin)
    dx = mzeros(input_size)
    map(opidxp->dx[opidxp[1]].=opidxp[2]/prod(kernel_size), zip(opidxs,y))
    return ongpu(dx)
end

@primitive avgpool(input, strides, dilations),dy,y avgpoolx(input, strides, dilations, y, dy)
@zerograd avgpoolx(input, strides, dilations, dy, y)

function kmaxvaluesv3(a::Array{T, 4}, k) where {T}
    a_size = size(a)
    vas = []
    for i in 1:a_size[1]:prod(a_size)
        va = mzeros(Int64, k)
        for avi in i:i+a_size[1]-1
            for vai in 1:k
                if va[vai] == 0 || a[avi] >= a[va[vai]]
                    if va[vai] != 0
                        insert!(va, vai, avi)
                        va = va[1:end-1]
                    else
                        va[vai] = avi
                    end
                    break
                end
            end
        end
        push!(vas, va)
    end
    return reshape(hcat(vas...), (k, a_size[2:end]...))
end
@zerograd kmaxvaluesv3(a,k)


function kmax_pf(x::Union{Array{T, 4}, AutoGrad.Result{Array{T,4}}}, k::Int) where {T}
    x = permutedims(x, (2,1,3,4))
    x = x[kmaxvaluesv3(x,k)]
    x = permutedims(x, (2,1,3,4))
    return x
end
