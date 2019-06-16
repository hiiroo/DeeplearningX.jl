#=

Copyright (c) 2019 Ali Mert Ceylan

=#

function ongpu(d)
  (haskey(Pkg.installed(), "CuArrays")) ? cu(d) : d
end


"
LayerTelemetry type. Store weight, bias, layer output shapes correspondingly w,b,o.
"
struct LayerTelemetry; w; b; o; end


"
t->Telemetry


"
struct ConvolutionLayer
  telemetry
  weight
  bias
  activation

  function ConvolutionLayer(k, s, d, init, act; w=nothing, b=nothing)

    function c(i)

        input_size = i[1:end-2]
        kernel_size = _kernelsize(k[1:end-1], d)
        fm_size = _featuremapsize(input_size, kernel_size, s)
        ws = (kernel_size...,i[3],k[3])
        bs = (1,1,k[3],1)
        os = (fm_size...,k[end],i[end])

        (cl::ConvolutionLayer) =  new(
          LayerTelemetry(ws, bs, os),
          w == nothing ? Param(ongpu(init(ws...))) : Param(ongpu(w)),
          b == nothing ? Param(ongpu(init(bs...))) : Param(ongpu(b)),
          act)

        return cl
    end
  end
end
(c::ConvolutionLayer)(x;args...) = c.activation.(conv(c.weight, x;args...) .+ c.bias)


"
t->Telemetry

"
struct FullyConnectedLayer
  telemetry
  weight
  bias
  activation

  function FullyConnectedLayer(n, init, act; w=nothing, b=nothing)

    function d(i)
        weight_size = (n, prod(i[1:end-1]))
        fm_size = (n, 1, i[end])
        return new(
          LayerTelemetry(weight_size, fm_size,  fm_size),
          w == nothing ? Param(ongpu(init(weight_size...))) : Param(ongpu(w)),
          b == nothing ? Param(ongpu(init(fm_size[1:end-1]...))) : Param(ongpu(b)),
          act)
    end
  end
end
(d::FullyConnectedLayer)(x) = d.activation.((d.weight*mat(x)).+d.bias)


"
t->Telemetry

"
mutable struct RecurrentLayer
  telemetry
  weight
  hweight
  bias
  hbias
  activation
  h
  f

  function RecurrentLayer(n, init, act; w=nothing, hw=nothing, b=nothing, hb=nothing)

    function r(i)
        weight_size = (n, prod(i[1:end-1]))
        hweight_size = (n, n) #weight_size
        fm_size = (n, 1, i[end])

        function rnn(rl::RecurrentLayer, x)
          a = (rl.weight*mat(x)).+(rl.hweight*mat(rl.h)).+rl.bias
          rl.h = rl.activation.(a)
          return rl.activation.(a.+rl.hbias)
        end

        return new(
          LayerTelemetry(weight_size, fm_size,  fm_size),
          w == nothing ? Param(ongpu(init(weight_size...))) : Param(ongpu(w)),
          hw == nothing ? Param(ongpu(init(hweight_size...))) : Param(ongpu(hw)),
          b == nothing ? Param(ongpu(init(fm_size[1:end-1]...))) : Param(ongpu(b)),
          hb == nothing ? Param(ongpu(init(fm_size[1:end-1]...))) : Param(ongpu(hb)),
          act,
          Param(ongpu(init(n, 1))),
          rnn)
    end
  end
end
(r::RecurrentLayer)(x) = r.f(r, x)


"
t->Telemetry

"
struct PoolLayer
  t
  f

  function PoolLayer(w,s,d,m)
    function p(i)
        input_size = i[1:end-2]
        kernel_size = _kernelsize(w, d)
        fm_size = _featuremapsize(input_size, kernel_size, s)
        os = (fm_size..., i[end-1:end]...)

        if(m==0)
          pfunc(x) = cat([cat([maxpool(x[:,:,xi,bi], w, s, d) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]...,dims=4)
        elseif(m==1)
          pfunc(x) = cat([cat([avgpool(x[:,:,xi,bi], w, s, d) for xi in 1:size(x)[3]]..., dims=3) for bi in 1:size(x)[4]]..., dims=4)
        else
          pfunc(x,k) = x[kmax(x,k)]
        end

        return new(LayerTelemetry(nothing, nothing, os), pfunc)
    end
  end
end
(p::PoolLayer)(x;args...) = p.f(x;args...)
Pooling(t, f) = PoolLayer(t, f)

#=
"
t->Telemetry

"
struct DropoutLayer; t; f; end
(d::DropoutLayer)(x) = d.f(x)
Dropping(t, f) = DropoutLayer(t, f)
=#
#=
"
t->Telemetry

fd->(f)old (d)imension i.e. 2
"
struct FoldingLayer; t; d; f; end
(f::FoldingLayer)(x) = f.f(x)
Folding(t, d, f) = FoldingLayer(t, d, f)
=#

"
layers->Array of layers

functn->Network as a function

lossfn->Loss function
"
mutable struct Network; layers; functn; lossfn; end
(n::Network)(x::AbstractArray) = n.functn(x)
(n::Network)(x::AbstractArray,y::AbstractArray;kwargs...) = n.lossfn(n(x),y;kwargs...)
(n::Network)(d::Data) = mean(n(x, y) for (x,y) in d)
Network(l,f;loss=nll) = Network(l,f,loss)

#=
function Dropout(p)
	function d(i)
		return Dropping(LayerTelemetry(nothing, nothing, i), x->dropout(x, p))
	end
end
=#
#=
"
Fold(d)

d: Fold coefficient

Usage:

julia> f1layer = Fold(2)((48, 7, 1, 1))

Expected output: f1layer.t.o=(24, 7, 1, 1)
"
function Fold(d)

    function f(i)
        input_size = i[1:2]
        fm_size = (input_size[1] != nothing ? Int(ceil(input_size[1]/d)) : nothing, input_size[2])
        os = (fm_size..., i[3:end]...)
        return Folding(LayerTelemetry(nothing, nothing, os), d, x->pool(x, window=(d,1), stride=(d,1), padding=(0,0), mode=1).*d)
    end
end
=#

"
t->Telemetry

d->Dict; for word and word vectors

f->Function; embedding function

Embed(s,d); Embedding layer.

s: Vector size i.e. 16 if word vectors have shape of (16,1)

d: Dict; dictionary aka lookup table for words to corresponding vectors

Usage:

If sentences have the same length they can be given as minibatches to layer.

e1layer = Embed(16, lookup_table)(11, 100)

Expected output: e1layer.t.o=(16,11,1,100)

In an online learning setup, using minibatches won't be possible since sentence lengths will not be same, for that case;

e1layer = Embed(16, lookup_table)()

Expected output: e1layer.t.o=(16,nothing,1,1)
"
struct EmbeddingLayer
  t
  d
  f
  function EmbeddingLayer(s, d;kwargs...)
    function e(i=nothing, b=nothing, dims=2)
    	function ef(xs; dims=dims)
    		return reshape(hcat([hcat([get(e.d, xi, tryparse(Float32, xi) != nothing ? e.d["<num>"] : e.d["<unk>"])  for xi in x]...) for x in xs]...), (s, dims==2 ? length(xs[1]) : 1, dims==3 ? length(xs[1]) : 1, dims==4 ? length(xs[1])*length(xs) : length(xs)))
    	end
        new(LayerTelemetry(nothing, nothing, (s, dims==2 ? i : 1, dims==3 ? i : 1, b !=nothing ? dims==4 ? i : b : 1)), d, ef)
    end
  end
end
(e::EmbeddingLayer)(x; kwargs...) = e.f(x; kwargs...)
