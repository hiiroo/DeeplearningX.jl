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
softmax(x) = cat([(exp10.(log.(x[:,bi])))./(sum(exp10.(log.(x[:,bi])))) for bi in 1:size(x)[end]]...,dims= 2)

#Loss functions
"Squared difference."
squared_diff(x, y) = sum(abs2.(x-y))