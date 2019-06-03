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

#Array Utilities

"
Function creator macro to enforce float32 type on random or other weight initialization functions that support rand(Float32, m, n, ...) format

Usage:

julia> xavierfloat32 = @float32 xavier

julia> xavierfloat32(4,5)

0.127307  -0.0146135  -0.278793  -0.11998    -0.446539
0.292935   0.0432366   0.305089  -0.0674677   0.299479
0.388128   0.244012    0.303878  -0.421147    0.0951707
0.139119  -0.402997    0.165534  -0.270187   -0.217053
"
macro float32(f)
    g(x...) = eval(Expr(:call, f, Float32, x...))
    g
end

macro createarray(sz)
    quote
        a = rand(Float32, $(esc(sz)))
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