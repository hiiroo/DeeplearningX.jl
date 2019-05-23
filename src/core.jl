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
Convolution operation. Used by @convolve macro to create a
convolution operation with specified input, kernel, stride
and dilation configuration.
"
mutable struct Convolution
	m
	k
	f
end

"
"
mutable struct Pooling
	f
end

"
"
mutable struct Densemul
	m
	f
end

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
function im2col(weight, input_size::Tuple{Int, Int}, kernel_size::Tuple{Int, Int}, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int}, fm_size::Tuple{Int, Int})
    margin = input_size.%fm_size
    ap=[]
    for i in 1:strides[1]:(fm_size[1]*strides[1])
        for j in 1:strides[2]:(fm_size[2]*strides[2])
            t1 = zeros(input_size)
            t1[i:dilations[1]:(i+kernel_size[1]-1), j:dilations[2]:(j+kernel_size[2]-1)] = weight'
            ap = prod(size(ap) .<= 0) ? t1'[:] : hcat(ap, t1'[:])
        end
    end

    return ap
end

"Fetches convolution kernel from given convolution matrix, input_size, kernel size, stride and dilation configuration"
function col2im(conv_matrix, input_size::Tuple{Int, Int}, weight_size::Tuple{Int, Int}, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int})
    kernel_size = _kernelsize(weight_size, dilations)

    _dimcheck_convolutionkernel(input_size, kernel_size)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    margin = _convolutionkernelmargin(input_size, kernel_size)

    cki = 1
    ck = zeros(weight_size)
    for i in (1+margin[1]):strides[1]:(input_size[1]-margin[2])
        for j in (1+margin[1]):strides[2]:(input_size[2]-margin[2])
            temp_ck = reshape(conv_matrix[:,cki], input_size)'[i-margin[1]:dilations[1]:(i+(margin[1])), j-margin[1]:dilations[2]:(j+(margin[1]))]
            ck .+= temp_ck'
            cki+=1
        end
    end

    return ck ./(size(conv_matrix)[2])
end


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
kmax(a, k::Int) = reshape([ce in sort(a[:,ci])[end-k:end] ? true : false for ci in 1:size(a)[2] for ce in a[:,ci]], size(a))
@zerograd kmax(a, k::Int)
