module Deeplearning
	using Pkg
	!(haskey(Pkg.installed(), "CuArrays")) || using CuArrays
	using AutoGrad
	using LinearAlgebra

	export @convolve, @dense, @maxpool, @kmaxpool, @cudaarray, @createarray, @parameters, @onehot, @onehotencode, @onehotdecode, sigmoid, relu, softmax, squared_diff

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

	function LogicalIndices(boolarr)
	    arr_size = size(boolarr)
	    c = reshape(collect(1:prod(size(boolarr))), (prod(size(boolarr)),1))
	    return c[reshape(boolarr, (prod(size(boolarr)),1))]
	end


	_dimcheck_convolutionkernel(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}) = prod(inputsize .> kernelsize) || throw(ArgumentError("Kernel size $kernelsize cannot be larger than Input size $inputsize"))

	_kernelsize(kernelsize::Tuple{Int, Int}, dilations::Tuple{Int, Int}) = ((dilations .* kernelsize) .- (dilations.-1))
	_featuremapsize(inputsize::Tuple{Int, Int}, kernelsize::Tuple{Int, Int}, strides::Tuple{Int, Int}) = div.(inputsize .- kernelsize, strides) .+ 1
	_convolutionkernelmargin(inputsize::Tuple{Int, Int}, featuremapsize::Tuple{Int, Int}) = Int.(ceil.((inputsize.-featuremapsize) ./ 2)).-(1 .*(featuremapsize.%2))


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
	    margin = _convolutionkernelmargin(input_size, fm_size)

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


	"Convolution operation. Used by @convolve macro to create a
	convolution operation with specified input, kernel, stride
	and dilation configuration"
	mutable struct Convolution
	    "Convolution matrix. @convolve macro creates the
	    convolution matrix with respect to the expected inputsize"
	    matrix
	    "Convolution kernel function. @convolve macro creates
	    a function to retrieve convolution kernels from convolution matrix"
	    kernel
	    f
	end


	"Convolve macro creates convolution matrix wrt expected input and specified
	stride and dilation configuration. Return a convolution struct with necessary
	specifications."
	macro convolve(i, k, s, d)
	    cop = quote
	        local i = $(esc(i))
	        local k = $(esc(k))
	        local s = $(esc(s))
	        local d = $(esc(d))

	        strides = length(s) < 2 ? (s, s) : s

	        dilations = length(d) < 2 ? (d, d) : d

	        input_size = size(i)[1:end-1]
	        kernel_size = _kernelsize(size(k)[1:end-1], dilations)

	        _dimcheck_convolutionkernel(input_size, kernel_size)
	        fm_size = _featuremapsize(input_size, kernel_size, strides)

	        t1 = cat([im2col(k[:,:,ki], input_size, kernel_size, strides, dilations, fm_size) for ki in 1:size(k)[3]]...,dims=3)
	        cut1 = @cudaarray permutedims(t1,(2,1,3))#transpose every convmat
	        cop = Convolution(Param(cut1), nothing, nothing)#, cu(zeros(prod(fm_size), size(k)[3]*size(i)[3]))
	        cop.kernel = ()->cat([col2im(collect(cop.matrix[:,:,ki])', input_size, size(k)[1:end-1], strides, dilations) for ki in 1:size(k)[3]]...,dims=3)

	        function convit(x)
	            x_channels = size(x)[3]
	            cm_channels = size(cop.matrix)[3]
	            reshaped_x = reshape(x, (prod(size(x)[1:2]), size(x)[3]))
	            return reshape(cat([cop.matrix[:,:,cmi]*reshaped_x for cmi in 1:cm_channels]..., dims=3), (fm_size..., cm_channels*x_channels))
	        end

	        cop.f = convit

	        cop
	    end
	    @eval function (cop::Convolution)(x)  cop.f(x) end
	    cop
	end

	#Pooling operations

	function pool(op::Function, input, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int})
	    input_size = size(input)
	    kernel_size = _kernelsize(strides, dilations)

	    _dimcheck_convolutionkernel(input_size, kernel_size)
	    fm_size = _featuremapsize(input_size, kernel_size, strides)
	    margin = input_size.%kernel_size

	    ap = zeros(input_size)
	    for i in (1):strides[2]:(input_size[2]-margin[2])
	        for j in (1):strides[1]:(input_size[1]-margin[1])
	            t1 = input[j:dilations[2]:(j+strides[2]-1), i:dilations[1]:(i+strides[1]-1)]
	            t1max = op(t1)
	            ap[(Tuple(t1max[2]).+(j,i).-(1, 1))...] = true
	        end
	    end

	    return BitArray(ap)
	end
	@zerograd pool(op::Function, input, strides::Tuple{Int, Int}, dilations::Tuple{Int, Int})

	"k-max operation, a is an Array and k is the maximum k element of the column"
	kmax(a, k::Int) = reshape([ce in sort(a[:,ci])[end-k:end] ? true : false for ci in 1:size(a)[2] for ce in a[:,ci]], size(a))
	@zerograd kmax(a, k::Int)

	""
	mutable struct Pooling
	    f
	end

	"
	2D max pooling operation. i as input, s as pool dimensions, d as dilations.
	Example: @maxpool rand(4,4) (2,2) (1,1)
	"
	macro maxpool(i, s, d)
	    poolop=quote
	        input = $(esc(i))
	        strides = $(esc(s))
	        dilations = $(esc(d))

	        input_size = size(input)[1:end-1]
	        kernel_size = _kernelsize(strides, dilations)

	        _dimcheck_convolutionkernel(input_size, kernel_size)
	        fm_size = _featuremapsize(input_size, kernel_size, strides)
	        poolop = Pooling(x->cat([reshape((x[:,:,xi])[LogicalIndices(pool(findmax, x[:,:,xi], strides, dilations))], (fm_size...,1)) for xi in 1:size(x)[3]]...,dims=3))
	        poolop
	    end
	    @eval function (poolop::Pooling)(x) poolop.f(x) end
	    poolop
	end

	"k-max pooling operation, creates another array with the size of k, n when given m, n array"
	macro kmaxpool(i, k)
	    poolop = quote
	        input = $(esc(i))
	        kval = $(esc(k))
	        poolop = Pooling(x->cat([(reshape((x[:,:,xi])[LogicalIndices(kmax(x[:,:,xi],kval))], (kval+1, size(x)[2]))')' for xi in 1:size(x)]..., dims=3))
	        poolop
	    end
	    @eval function (poolop::Pooling)(x) poolop.f(x) end
	    poolop
	end

	mutable struct Densemul
	    matrix
	    f
	end

	macro dense(i, n)
	    denseop = quote
	        ei = $(esc(i))
	        en = $(esc(n))
	        w = @cudaarray rand(en, prod(size(ei)))
	        denseop = Densemul(Param(w), nothing)
	        denseop.f = x->denseop.matrix*reshape(x,(prod(size(ei)),1))
	        denseop
	    end
	    @eval function (denseop::Densemul)(x) denseop.f(x) end
	    denseop
	end

	#Activation functions
	"Sigmoid activation function."
	sigmoid(x) = 1 ./(1 .+exp.(-x))

	relumap(x) = Int.(map(x->x>0, x))
	@zerograd relumap(x::Any)

	"ReLU activation function."
	function relu(x)
	    return x.*relumap(x)
	end


	"Softmax activation function."
	softmax(x) = (exp10.(log.(x)))./(sum(exp10.(log.(x))))#(exp.(x))./sum(exp.(x))

	#Loss functions
	"Squared difference."
	squared_diff(x, y) = sum(abs2.(x-y))

	#Optimization Utilities

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


end # module
