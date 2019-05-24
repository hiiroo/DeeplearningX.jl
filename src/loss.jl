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

#Loss functions
"
Squared difference
"
squared_diff(x, y) = sum(abs2.(x-y))

"
Negative Log Likelihood loss

p_y: 2D matrix of outputs

y: Array of correct indices
"
function nll(p_y, y; average=true)
    lp = [p_y[y[p_s],p_s] for p_s in 1:size(p_y)[2]]
    average ? mean(lp) : sum(lp)
end


"
Classification accuracy

p_y: 2D matrix of outputs

y: Array of correct indices
"
function acc(p_y, y)
    return count([ci[1]==y[1] for (cii,ci) in enumerate(findmax(p_y,dims=1)[2])])/size(p_y)[2]
end