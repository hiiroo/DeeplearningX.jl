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

module Deeplearning
	using Pkg
	!(haskey(Pkg.installed(), "CUDAnative")) || using CUDAnative
	!(haskey(Pkg.installed(), "CUDAdrv")) || using CUDAdrv
	!(haskey(Pkg.installed(), "CuArrays")) || using CuArrays
	using AutoGrad
	using LinearAlgebra
	using LinearAlgebra.BLAS
	using Statistics

	__ongpu = (haskey(Pkg.installed(), "CuArrays")) ? 1 : -1

	function gpu()
		global __ongpu
		return __ongpu
	end

	function gpu(d::Int)
		global __ongpu
		(haskey(Pkg.installed(), "CuArrays")) ? __ongpu = d : __ongpu = -1
		return __ongpu
	end

	!(haskey(Pkg.installed(), "CUDAnative")) || !(haskey(Pkg.installed(), "CUDAdrv")) || !(haskey(Pkg.installed(), "CuArrays")) || include("kernel.jl")

	include("macros.jl")
	include("dist.jl")
	include("data.jl")
	include("progress.jl")
	include("core.jl")
	include("interface.jl")
	include("act.jl")
	include("loss.jl")

	export 	@cudaarray,
			@createarray,
			@parameters,
			@onehot,
			@onehotencode,
			@onehotdecode,
			mat,
			gaussian,
			xavier,
			bilinear,
			sigmoid,
			relu,
			softmax,
			squared_diff,
			nll,
			acc,
			LayerTelemetry,
			ConvolutionLayer
			FullyConnectedLayer,
			PoolLayer,
			Network,
			Data,
			minibatch,
			repeat,
			progress

	mat(x) = reshape(x,(prod(size(x)[1:end-1]),size(x)[end]))
	blas_threads(x::Int) = LinearAlgebra.BLAS.set_num_threads(x)

end # module
