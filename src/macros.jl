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

# "Convolve macro creates convolution matrix wrt expected input and specified
# stride and dilation configuration. Return a convolution struct with necessary
# specifications."
# macro convolve(i, k, s, d)
#     cop = quote
#         local i = $(esc(i))
#         local k = $(esc(k))
#         local s = $(esc(s))
#         local d = $(esc(d))

# 		strides = length(s) < 2 ? (s, s) : s

# 		dilations = length(d) < 2 ? (d, d) : d

# 		input_size = size(i)[1:2]
# 		kernel_size = _kernelsize(size(k)[1:2], dilations)

# 		_dimcheck_convolutionkernel(input_size, kernel_size)
# 		fm_size = _featuremapsize(input_size, kernel_size, strides)

# 		t1 = cat([cat([im2col(k[:,:,ki,ci], input_size, kernel_size, strides, dilations, fm_size) for ki in 1:size(k)[3]]...,dims=3) for ci in 1:size(k)[4]]...,dims=4)
# 		cut1 = permutedims(t1,(2,1,3,4))
# 		cop = Convolution(cut1, nothing, nothing)
# 		cop.kernel = ()->cat([cat([col2im(collect(cop.matrix[:,:,ki,ci])', input_size, size(k)[1:2], strides, dilations) for ki in 1:size(k)[3]]...,dims=3) for ci in 1:size(k)[4]]...,dims=4)

# 		function convit(x)
# 		    x_size = size(x)
# 		    x_channels = x_size[3]
# 		    x_batch = x_size[4]

# 		    cm_size = size(cop.matrix)
# 		    cm_channels = cm_size[4]

# 		    reshaped_x = reshape(x, (prod(size(x)[1:3]), x_batch))
# 		    return reshape(cat([reshape(reshape(cop.matrix[:,:,:,cmi], (cm_size[1], prod(cm_size[2:3])))*reshaped_x, (prod(fm_size), 1, x_batch)) for cmi in 1:cm_channels]...,dims=2), (fm_size..., cm_channels, x_batch))
# 		end

#         cop.f = convit

#         cop
#     end
#     @eval function (cop::Convolution)(x)  cop.f(x) end
#     cop
# end


# "
# 2D max pooling operation. i as input, s as pool dimensions, d as dilations.
# Example: @maxpool rand(4,4) (2,2) (1,1)
# "
# macro maxpool(i, w, s, d)
#     poolop=quote
#         input = $(esc(i))
#         window = $(esc(w))
#         strides = $(esc(s))
#         dilations = $(esc(d))

#         input_size = size(input)[1:2]
#         kernel_size = _kernelsize(strides, dilations)

#         _dimcheck_convolutionkernel(input_size, kernel_size)
#         fm_size = _featuremapsize(input_size, kernel_size, strides)
#         poolop = Pooling(x->cat([cat([maxpool(x[:,:,xi,bi], window, strides, dilations) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]...,dims=4))
#         poolop
#     end
#     @eval function (poolop::Pooling)(x) poolop.f(x) end
#     poolop
# end

# "
# 2D avg pooling operation. i as input, s as pool dimensions, d as dilations.
# Example: @avgpool rand(4,4) (2,2) (1,1)
# "
# macro avgpool(i, w, s, d)
#     poolop=quote
#         input = $(esc(i))
#         window = $(esc(w))
#         strides = $(esc(s))
#         dilations = $(esc(d))

#         input_size = size(input)[1:2]
#         kernel_size = _kernelsize(strides, dilations)

#         _dimcheck_convolutionkernel(input_size, kernel_size)
#         fm_size = _featuremapsize(input_size, kernel_size, strides)
#         poolop = Pooling(x->cat([cat([avgpool(x[:,:,xi,bi], window, strides, dilations) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]..., dims=4))
#         poolop
#     end
#     @eval function (poolop::Pooling)(x) poolop.f(x) end
#     poolop
# end

# "k-max pooling operation, creates another array with the size of k, n when given m, n array"
# macro kmaxpool(i, k)
#     poolop = quote
#         input = $(esc(i))
#         kval = $(esc(k))
#         poolop = Pooling(x->cat([(reshape((x[:,:,xi])[LogicalIndices(kmax(x[:,:,xi],kval))], (kval+1, size(x)[2]))')' for xi in 1:size(x)]..., dims=3))
#         poolop
#     end
#     @eval function (poolop::Pooling)(x) poolop.f(x) end
#     poolop
# end

# macro dense(i, n)
#     denseop = quote
#         ei = $(esc(i))
#         en = $(esc(n))
#         w = @cudaarray rand(en, prod(size(ei)[1:end-1]))
#         denseop = Densemul(Param(w), nothing)
#         denseop.f = x->denseop.matrix*reshape(x,(prod(size(x)[1:end-1]),size(x)[end]))
#         denseop
#     end
#     @eval function (denseop::Densemul)(x) denseop.f(x) end
#     denseop
# end

#Array Utilities

macro createarray(sz)
    quote
        a = rand(Float64, $(esc(sz)))
        (haskey(Pkg.installed(), "CuArrays")) ? cu(a) : a
    end
end

macro cudaarray(arr)
    quote
        (haskey(Pkg.installed(), "CuArrays")) ? cu($(esc(arr))) : $(esc(arr))
    end
end

#AutoGrad specific

macro parameters(fvals)
    quote
        collect(params($(esc(fvals))))
    end
end


#Data Utilities

macro onehot(l)
    quote
        array = collect(Set($(esc(l))))
        one_hots = zeros(length(array), length(array))
        one_hots[diagind(one_hots)].=1
        d = Dict([array[i] => @cudaarray one_hots[1:end,i] for i in 1:size(one_hots)[2]])
        dr = Dict(Int.(collect(value)) => key for (key, value) in d)
        (d, dr)
    end
end

macro onehotencode(d, l)
    quote
        $(esc(d[l]))
    end
end

function decodeonehot(dr, o)
    outkey = collect(Int.(o.==maximum(o)))[:]
    outkey in keys(dr) ? dr[outkey] : nothing
end

macro onehotdecode(dr, o)
    quote
        decodeonehot($(esc(dr)), $(esc(o)))
        # $(esc(dr))[collect(Int.($(esc(o)).==maximum($(esc(o)))))[:]]
    end
end