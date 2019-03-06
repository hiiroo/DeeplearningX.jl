module PoolingTests
        using Pkg
        !(haskey(Pkg.installed(), "CuArrays")) || using CuArrays
        using Deeplearning
        using Test

        i1 = [  1. 1. 1. 1. 1. 2. 2. 2. 2. 2.;
                1. 1. 1. 1. 1. 2. 2. 2. 2. 2.;
                1. 1. 1. 1. 1. 2. 2. 2. 2. 2.;
                1. 1. 1. 1. 1. 2. 2. 2. 2. 2.;
                1. 1. 1. 1. 1. 2. 2. 2. 2. 2.;
                3. 3. 3. 3. 3. 4. 4. 4. 4. 4.;
                3. 3. 3. 3. 3. 4. 4. 4. 4. 4.;
                3. 3. 3. 3. 3. 4. 4. 4. 4. 4.;
                3. 3. 3. 3. 3. 4. 4. 4. 4. 4.;
                3. 3. 3. 3. 3. 4. 4. 4. 4. 4.   ]

        i1 = i1'
        i1 = reshape(i1, (10,10,1))
        i1 = @cudaarray i1

        function maxpool_1()
             eo = [     1. 1. 2. 2. 2.;
                        1. 1. 2. 2. 2.;
                        3. 3. 4. 4. 4.;
                        3. 3. 4. 4. 4.;
                        3. 3. 4. 4. 4.      ]

                pool = @maxpool i1 (2,2) (2,2) (1,1)
                @test pool(i1)[:,:,1]' == eo
        end

        function avgpool_1()
             eo = [     1.0  1.0  1.5  2.0  2.0
                        1.0  1.0  1.5  2.0  2.0
                        2.0  2.0  2.5  3.0  3.0
                        3.0  3.0  3.5  4.0  4.0
                        3.0  3.0  3.5  4.0  4.0     ]

                pool = @avgpool i1 (2,2) (2,2) (1,1)
                @test pool(i1)[:,:,1]' == eo
        end

        @testset "maxpool" begin
                maxpool_1()
        end

        @testset "avgpool" begin
                avgpool_1()
        end
end
