module ConvolutionTests
        using Pkg; !(haskey(Pkg.installed(), "CuArrays")) || using CuArrays
        using Deeplearning
        using Test


        w = [1 2 3 ; 4 5 6 ; 7 8 9]
        w = w'
        w = reshape(w, (3,3,1,1))

        i1 = [  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
                0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0    ]

        i1 = i1'
        i1 = reshape(i1, (10,10,1,1))
        i1 = @cudaarray i1

        function mytestfunction()
                eo = [  0.  0.  0.  0.  0.  0.  0.  0.;
                        0.  0.  9. 17. 24. 15.  7.  0.;
                        0.  0. 15. 28. 39. 24. 11.  0.;
                        0.  0. 18. 33. 45. 27. 12.  0.;
                        0.  0.  9. 16. 21. 12.  5.  0.;
                        0.  0.  3.  5.  6.  3.  1.  0.;
                        0.  0.  0.  0.  0.  0.  0.  0.;
                        0.  0.  0.  0.  0.  0.  0.  0.  ]

                
                @test Deeplearning.conv(w,i1,s=(1,1),d=(1,1))[:,:,1,1]' == eo
        end

        function mytestfunction2()
                eo = [  9.  9.  17.  8.  15. 7. ;
                        15. 15. 28.  13. 24. 11.;
                        6.  6.  11.  5.  9.  4. ;
                        9.  9.  16.  7.  12. 5. ;
                        3.  3.  5.   2.  3.  1. ;
                        3.  3.  5.   2.  3.  1.  ]

                # conv = @convolve i1 w (1,1) (2,2)
                # @test conv(i1)[:,:,1]' == eo
                @test Deeplearning.conv(w,i1,s=(1,1),d=(2,2))[:,:,1,1]' == eo
        end

        function mytestfunction3()
                eo = [  6. 5. 5. 5.;
                        6. 5. 5. 5.;
                        6. 5. 5. 5.;
                        3. 2. 2. 2.  ]

                # conv = @convolve i1 w (1,1) (3,3)
                # @test conv(i1)[:,:,1,1]' == eo
                @test Deeplearning.conv(w,i1,s=(1,1),d=(3,3))[:,:,1,1]' == eo
        end

        function mytestfunction4()
                eo = [  5. 5.;
                        5. 5. ]

                # conv = @convolve i1 w (1,1) (4,4)
                # @test conv(i1)[:,:,1,1]' == eo
                @test Deeplearning.conv(w,i1,s=(1,1),d=(4,4))[:,:,1,1]' == eo
        end

        @testset "dilated_kernel_tests" begin
                mytestfunction()
                mytestfunction2()
                mytestfunction3()
                mytestfunction4()
        end
end
