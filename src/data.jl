#=
The Knet.jl package is licensed under the MIT "Expat" License:

> Copyright (c) 2015: Deniz Yuret.
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

# Modified for Deeplearning by Ali Mert Ceylan


using Random
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle

mutable struct Data{T}; x; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; xtype; ytype; end

"""
    minibatch(x, [y], batchsize; shuffle, partial, xtype, ytype, xsize, ysize)

Return an iterator of minibatches [(xi,yi)...] given data tensors x, y and batchsize.  

The last dimension of x and y give the number of instances and should be equal. `y` is
optional, if omitted a sequence of `xi` will be generated rather than `(xi,yi)` tuples.  Use
`repeat(d,n)` for multiple epochs, `Iterators.take(d,n)` for a partial epoch, and
`Iterators.cycle(d)` to cycle through the data forever (this can be used with `converge`).
If you need the iterator to continue from its last position when stopped early (e.g. by a
break in a for loop), use `Iterators.Stateful(d)` (by default the iterator would restart
from the beginning).

Keyword arguments:

- `shuffle=false`: Shuffle the instances every epoch.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.
- `ytype=typeof(y)`: Convert yi in minibatches to this type.
- `xsize=size(x)`: Convert xi in minibatches to this shape.
- `ysize=size(y)`: Convert yi in minibatches to this shape.
"""
minibatch, Data

function minibatch(x,y,batchsize; shuffle=false,partial=false,xtype=typeof(x),ytype=typeof(y),xsize=size(x), ysize=size(y))
    nx = size(x)[end]
    if nx != size(y)[end]; throw(DimensionMismatch()); end
    x2 = reshape(x, :, nx)
    y2 = reshape(y, :, nx)
    imax = partial ? nx : nx - batchsize + 1
    # xtype,ytype may be underspecified, here we infer the exact types from the first batch:
    ids = 1:min(nx,batchsize)
    xt = typeof(convert(xtype, reshape(x2[:,ids],xsize[1:end-1]...,length(ids))))
    yt = typeof(convert(ytype, reshape(y2[:,ids],ysize[1:end-1]...,length(ids))))
    Data{Tuple{xt,yt}}(x2,y2,batchsize,nx,partial,imax,1:nx,shuffle,xsize,ysize,xtype,ytype)
end

function minibatch(x,batchsize; shuffle=false,partial=false,xtype=typeof(x),xsize=size(x))
    nx = size(x)[end]
    x2 = reshape(x, :, nx)
    imax = partial ? nx : nx - batchsize + 1
    # xtype may be underspecified, here we infer the exact types from the first batch:
    ids = 1:min(nx,batchsize)
    xt = typeof(convert(xtype, reshape(x2[:,ids],xsize[1:end-1]...,length(ids))))
    Data{xt}(x2,nothing,batchsize,nx,partial,imax,1:nx,shuffle,xsize,nothing,xtype,nothing)
end

@propagate_inbounds function iterate(d::Data, i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        d.indices = randperm(d.length)
    end
    nexti = min(i + d.batchsize, d.length)
    ids = d.indices[i+1:nexti]
    xbatch = convert(d.xtype, reshape(d.x[:,ids],d.xsize[1:end-1]...,length(ids)))
    if d.y == nothing
        return (xbatch,nexti)
    else
        ybatch = convert(d.ytype, reshape(d.y[:,ids],d.ysize[1:end-1]...,length(ids)))
        return ((xbatch,ybatch),nexti)
    end
end

eltype(::Type{Data{T}}) where T = T

function length(d::Data)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function rand(d::Data)
    i = rand(0:(d.length-d.batchsize))
    return iterate(d, i)[1]
end

# Use repeat(data,n) for multiple epochs; cycle(data) to go forever, take(data,n) for partial epochs

struct Repeat; data::Data; n::Int; end

repeat(d::Data, n::Int) = (@assert n >= 0; Repeat(d,n))
length(r::Repeat) = r.n * length(r.data)
eltype(r::Repeat) = eltype(r.data)
eltype(c::Cycle{Data}) = eltype(c.xs)
eltype(c::Cycle{Repeat}) = eltype(c.xs)

@propagate_inbounds function iterate(r::Repeat, s=(1,))
    epoch, state = s[1], tail(s)
    epoch > r.n && return nothing
    next = iterate(r.data, state...)
    next === nothing && return iterate(r, (epoch+1,))
    (next[1], (epoch, next[2]))
end

# Give length info in summary:
Base.summary(d::Data) = "$(length(d))-element $(typeof(d))"
