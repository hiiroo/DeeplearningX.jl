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


_dimcheck_convolutionkernel(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}) = prod(inputsize .> kernelsize) || throw(ArgumentError("Kernel size $kernelsize cannot be larger than Input size $inputsize"))
_kernelsize(kernelsize::Tuple{Int, Int}, dilations::Tuple{Int, Int}) = ((dilations .* kernelsize) .- (dilations.-1))
_featuremapsize(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}, strides::Tuple{Int, Int}) = div.(inputsize .- kernelsize, strides) .+ 1
_convolutionkernelmargin(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}) = Int.(ceil.(inputsize .% kernelsize))


"Creates convolution matrix according to given input, kernel, stride and dilation configuration"
function im2col(conv_matrix, conv_kernel, input_size::Tuple{Int, Int}, kernel_size::Tuple{Int, Int}, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int}, fm_size::Tuple{Int, Int})
    margin = input_size.%fm_size
    conv_matrix.=0
    t1 = zeros(input_size)
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
function col2im(conv_matrix, conv_kernel, input_size::Tuple{Int, Int}, weight_size::Tuple{Int, Int}, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int}, fm_size::Tuple{Int, Int})
    kernel_size = size(conv_kernel)
    margin = _convolutionkernelmargin(input_size, kernel_size[1:2])#

    conv_kernel .= 0
    for ci in 1:kernel_size[4]
        for ki in 1:kernel_size[3]
            cki = 1
            for i in (1+margin[1]):strides[1]:(input_size[1]-margin[2])
                for j in (1+margin[1]):strides[2]:(input_size[2]-margin[2])
                    temp_ck = reshape(conv_matrix[cki,:,ki,ci], input_size)'[i-margin[1]:dilations[1]:(i+(margin[1])), j-margin[1]:dilations[2]:(j+(margin[1]))]
                    conv_kernel[:,:,ki,ci].+=temp_ck'
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
    conv_mat=zeros(cm_size...)

    reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))

    im2col(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)

    return reshape(cat([reshape(reshape(conv_mat[:,:,:,cmi], (cm_size[1], prod(cm_size[2:3])))*reshaped_x, (prod(fm_size), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (fm_size..., cm_channels, x_batch))
end


function convx(w, x, dy;s=(1,1),d=(1,1))
    x_size = size(x)
    x_channels = x_size[3]
    x_batch = x_size[4]

    kernel_size = _kernelsize(size(w)[1:2], d)
    fm_size = _featuremapsize(x_size[1:2], kernel_size, s)

    cm_size = (prod(fm_size), prod(x_size[1:2]), size(w)[3:end]...)
    cm_channels = cm_size[4]
    conv_mat=zeros(cm_size...)

    im2col(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)
    conv_mat = permutedims(conv_mat, (2,1,3,4))
    conv_mat = permutedims(conv_mat, (1,2,4,3))
    cm_size = size(conv_mat)
    cm_channels = cm_size[4]

    dy_size = size(dy)
    dy_batch=dy_size[4]
    reshaped_dy = reshape(dy, (prod(dy_size[1:3]), dy_batch))

    return reshape(cat([reshape(reshape(conv_mat[:,:,:,cmi], (cm_size[1], prod(cm_size[2:3])))*reshaped_dy, (prod(x_size[1:2]), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (x_size[1:2]..., cm_channels, x_batch))
end


function convw(w, x, dy;s=(1,1),d=(1,1))
    x_size = size(x)
    x_channels = x_size[3]
    x_batch = x_size[4]

    kernel_size = _kernelsize(size(w)[1:2], d)
    fm_size = _featuremapsize(x_size[1:2], kernel_size, s)

    cm_size = (prod(fm_size), prod(x_size[1:2]), size(w)[3:end]...)
    cm_channels = cm_size[4]
    conv_mat=zeros(cm_size...)

    dy = permutedims(dy, (1,2,4,3))
    dy_size = size(dy)
    dy_batch=dy_size[3]
    dy_channels=dy_size[4]
    reshaped_dy = reshape(dy, (prod(size(dy)[1:2]), dy_batch, dy_channels))

    reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))'
    im2col(conv_mat, w, x_size[1:2], kernel_size, s, d, fm_size)

    dw = cat([reshape(reshape(reshaped_dy[:,:,dmi], (prod(dy_size[1:2]), dy_batch))*reshaped_x, (prod(dy_size[1:2]), prod(x_size[1:2]), x_channels, 1)) for dmi in 1:dy_channels]...,dims=4)

    return col2im(dw, w, x_size[1:2], size(w)[1:2], s, d, fm_size)
end


@primitive conv(w,x;args...),dy convx(w,x,dy;args...) convw(w,x,dy;args...) # maxpoolx(input, window, strides, dilations, y, dy)
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

    y = reshape(map(opidx->maximum(input[opidx]), poolidxs(findmaxidxs, input, window, strides, dilations, input_size, kernel_size, margin)), fm_size)
    return @cudaarray y
end

function maxpoolx(input, window, strides, dilations, dy, y)
    input_size = size(input)
    kernel_size = _kernelsize(strides, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    opidxs = poolidxs(findmaxidxs, input, window, strides, dilations, input_size, kernel_size, margin)
    dx = zeros(input_size)
    map(opidxp->dx[opidxp[1]]=opidxp[2], zip(opidxs,y))
    return @cudaarray dx
end

@primitive maxpool(input, window, strides, dilations),dy,y maxpoolx(input, window, strides, dilations, y, dy)
@zerograd maxpoolx(input, window, strides, dilations, dy, y)

function avgpool(input, window, strides, dilations)
    input_size = size(input)
    kernel_size = _kernelsize(window, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    y = reshape(map(opidx->mean(input[opidx]), poolidxs(findmeanidxs, input, window, strides, dilations, input_size, kernel_size, margin)), fm_size)
    return @cudaarray y
end

function avgpoolx(input, window, strides, dilations, dy, y)
    input_size = size(input)
    kernel_size = _kernelsize(window, dilations)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = input_size.%kernel_size

    opidxs = poolidxs(findmeanidxs, input, window, strides, dilations, input_size, kernel_size, margin)
    dx = zeros(input_size)
    map(opidxp->dx[opidxp[1]].=opidxp[2]/prod(kernel_size), zip(opidxs,y))
    return @cudaarray dx
end

@primitive avgpool(input, strides, dilations),dy,y avgpoolx(input, strides, dilations, y, dy)
@zerograd avgpoolx(input, strides, dilations, dy, y)


"k-max operation, a is an Array and k is the maximum k element of the column"
function kmax(a, k)
    a_size = size(a)
    a_cart = CartesianIndices(a)
    d_vtbs_rows = cu(zeros(a_size[1],a_size[1]))
    d_vtbs_cols = cu(zeros(a_size[1],a_size[1]))
    d_c = cu(zeros(Float32, a_size[1],a_size[1]))
    d_d = cu(zeros(a_size[1],1))
    out_array = Array{CartesianIndex}(undef, k,a_size[2:end]...)

    indices = collect(enumerate(1:a_size[1]:prod(a_size)))
    for (ii, i) in indices
        d_vtbs = a[i:i+a_size[1]-1]
        i_vtbs = a_cart[i:i+a_size[1]-1]

        @cuda threads=a_size[1] brows(d_vtbs_rows, d_vtbs, a_size[1])
        @cuda threads=a_size[1] bcols(d_vtbs_cols, d_vtbs, a_size[1])
        @cuda threads=a_size[1]^2 vcomp(d_c, d_vtbs_rows, d_vtbs_cols)
        @cuda threads=a_size[1] csum(d_d, d_c, a_size[1])
        d_d = a_size[1] .- d_d
        d_o = Array(d_d)
        ind = 1 .<= d_o .<=k
        out_array[(ii-1)*k+1:(ii-1)*k+k] .= i_vtbs[ind[:]]

        d_vtbs_rows.=0
        d_vtbs_cols.=0
        d_c.=0
        d_d.=0
    end

    return CartesianIndices(out_array)
end
@zerograd kmax(a, k::Int)

# "k-max operation, a is an Array and k is the maximum k element of the column"
# kmax(a, k::Int) = reshape([ce in sort(a[:,ci])[end-k:end] ? true : false for ci in 1:size(a)[2] for ce in a[:,ci]], size(a))


