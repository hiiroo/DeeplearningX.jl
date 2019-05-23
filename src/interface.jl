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

#Core Interface

"Convolve macro creates convolution matrix wrt expected input and specified
stride and dilation configuration. Return a convolution struct with necessary
specifications."
function convolve(i, k, s, d)
    strides = length(s) < 2 ? (s, s) : s
    dilations = length(d) < 2 ? (d, d) : d

    input_size = size(i)[1:2]
    kernel_size = _kernelsize(size(k)[1:2], dilations)

    _dimcheck_convolutionkernel(input_size, kernel_size)
    fm_size = _featuremapsize(input_size, kernel_size, strides)

    cop = Convolution(nothing, nothing, nothing)
    cm = permutedims(cat([cat([im2col(k[:,:,ki,ci], input_size, kernel_size, strides, dilations, fm_size) for ki in 1:size(k)[3]]...,dims=3) for ci in 1:size(k)[4]]...,dims=4),(2,1,3,4))

    function kern()
      cat([cat([col2im(collect(cop.m[:,:,ki,ci])', input_size, size(k)[1:2], strides, dilations) for ki in 1:size(k)[3]]...,dims=3) for ci in 1:size(k)[4]]...,dims=4)
    end

    function conv(x)
        x_size = size(x)
        x_channels = x_size[3]
        x_batch = x_size[4]

        cm_size = size(cop.m)
        cm_channels = cm_size[4]

        reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))
        return reshape(cat([reshape(reshape(cop.m[:,:,:,cmi], (cm_size[1], prod(cm_size[2:3])))*reshaped_x, (prod(fm_size), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (fm_size..., cm_channels, x_batch))
    end

    cop.m = cm
    cop.k = kern
    cop.f = conv
    @eval function (cop::Convolution)(x)  cop.f(x) end
    cop
end


"
2D max pooling operation. i as input, s as pool dimensions, d as dilations.
Example: @maxpool rand(4,4) (2,2) (1,1)
"
function maxpooling(i, w, s, d)
    input = i
    window = w
    strides = s
    dilations = d

    input_size = size(input)[1:2]
    kernel_size = _kernelsize(strides, dilations)

    _dimcheck_convolutionkernel(input_size, kernel_size)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    poolop = Pooling(x->cat([cat([maxpool(x[:,:,xi,bi], window, strides, dilations) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]...,dims=4))

    @eval function (poolop::Pooling)(x) poolop.f(x) end
    poolop
end

"
2D avg pooling operation. i as input, s as pool dimensions, d as dilations.
Example: @avgpool rand(4,4) (2,2) (1,1)
"
function avgpooling(i, w, s, d)

    input = i
    window = w
    strides = s
    dilations = d

    input_size = size(input)[1:2]
    kernel_size = _kernelsize(strides, dilations)

    _dimcheck_convolutionkernel(input_size, kernel_size)
    fm_size = _featuremapsize(input_size, kernel_size, strides)
    poolop = Pooling(x->cat([cat([avgpool(x[:,:,xi,bi], window, strides, dilations) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]..., dims=4))

    @eval function (poolop::Pooling)(x) poolop.f(x) end
    poolop
end

"k-max pooling operation, creates another array with the size of k, n when given m, n array"
function kmaxpooling(i, k)

    input = i
    kval = k
    poolop = Pooling(x->cat([(reshape((x[:,:,xi])[LogicalIndices(kmax(x[:,:,xi],kval))], (kval+1, size(x)[2]))')' for xi in 1:size(x)]..., dims=3))

    @eval function (poolop::Pooling)(x) poolop.f(x) end
    poolop
end

function dense(i, n)

    ei = i
    en = n
    w = @cudaarray rand(en, prod(size(ei)[1:end-1]))
    denseop = Densemul(Param(w), nothing)
    denseop.f = x->denseop.matrix*reshape(x,(prod(size(x)[1:end-1]),size(x)[end]))

    @eval function (denseop::Densemul)(x) denseop.f(x) end
    denseop
end
