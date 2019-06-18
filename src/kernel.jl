#=

Copyright (c) 2019 Ali Mert Ceylan

=#

function vcomp(c, a, b)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] > b[i] ? 1.0 : 0.0
    return
end

function csum(o, a, threadnum)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= threadnum
        for e in 1:threadnum
            o[i] += a[(e-1)*threadnum+i]
        end
    end
end

function brows(o, a, threadnum)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i<=threadnum)
        for e in 1:threadnum
            o[(e-1)*threadnum+i] = a[i]
        end
    end
end

function bcols(o, a, threadnum)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if(i<=threadnum)
        for e in 1:threadnum
            o[e+(i-1)*threadnum] = a[i]
        end
    end
end

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


function kmax_pf(x::Union{CuArray{T, 4}, AutoGrad.Result{CuArray{T,4}}}, k::Int) where {T}
    return x[kmax(x,k)]
end