
using Deeplearning

w1 = rand(3,3,4)
w2 = rand(3,3,3)

i = rand(20,20,1)
i = @cudaarray permutedims(i, (2,1,3))

conv1 = @convolve i w1 (1,1) (1,1)
pool1 = @maxpool conv1(i) (2,2) (1,1)
conv2 = @convolve pool1(conv1(i)) w2 (1,1) (1,1)
pool2 = @maxpool conv2(pool1(conv1(i))) (2,2) (1,1)
f(x) = pool2(conv2(pool1(conv1(x))))

o1 = f(i)

conv1.matrix
