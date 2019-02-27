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

        function mytestfunction()
             eo = [     1. 1. 2. 2. 2.;
                        1. 1. 2. 2. 2.;
                        3. 3. 4. 4. 4.;
                        3. 3. 4. 4. 4.;
                        3. 3. 4. 4. 4.      ]

                pool = @maxpool i1 (2,2) (1,1)
                @test pool(i1)[:,:,1]' == eo
        end
        

        @testset "pooling_tests" begin
                mytestfunction()
        end
end
