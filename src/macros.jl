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