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