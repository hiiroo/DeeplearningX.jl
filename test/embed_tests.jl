module EmbedTests
    using Pkg
    using AutoGrad
    using Deeplearning
    using BenchmarkTools
    using Test

    function embed_catview_1()
        words = ["this", "is", "a", "test", "<unk>", "<num>"]
        d = Dict(word => Param(rand(8, 1)) for word in words)
        
        e1layer = Deeplearning.EmbeddingLayer(8, d)(4, 1)
        f1layer = Deeplearning.FullyConnectedLayer(2, Deeplearning.xavier, Deeplearning.relu)(e1layer.telemetry.o)

        function f(n, x)
            x = e1layer(x, dims=2)
            x = f1layer(x)
            return x
        end
        
        n = Deeplearning.Network([e1layer, f1layer], f)
        # @show n([["this", "is", "a", "test"]])

        dv = @diff n([["this", "is", "a", "test"]], [0x01])
        # @show collect(AutoGrad.params(dv))
        @test dv != nothing            
    end

    function embed_catview_2()
        words = ["this", "is", "a", "test", "<unk>", "<num>"]
        d = Dict(word => Param(rand(8, 1)) for word in words)
        
        e1layer = Deeplearning.EmbeddingLayer(8, d)(22, 1)
        f1layer = Deeplearning.FullyConnectedLayer(2, Deeplearning.xavier, Deeplearning.relu)(e1layer.telemetry.o)

        function f(n, x)
            x = e1layer(x, dims=2)
            x = f1layer(x)
            return x
        end
        
        n = Deeplearning.Network([e1layer, f1layer], f)
        sent = ["this", "is", "a", "test", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>"]

        # @show n([sent])

        # @show @time @diff n([sent], [0x01])
        @test true
    end

    function embed_catview_3()
        words = ["this", "is", "a", "test", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>"]
        d = Dict(word => Param(rand(512, 1)) for word in words)
        
        e1layer = Deeplearning.EmbeddingLayer(512, d)(22, 1)
        f1layer = Deeplearning.FullyConnectedLayer(2, Deeplearning.xavier, Deeplearning.relu)(e1layer.telemetry.o)

        function f(n, x)
            x = e1layer(x, dims=2)
            x = f1layer(x)
            return x
        end
        
        n = Deeplearning.Network([e1layer, f1layer], f)
        sent = ["this", "is", "a", "test", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>", "this", "is", "a", "test", "<unk>", "<num>"]
        # @show n([sent])

        # @show @time @diff n([sent], [0x01])
        @test true
    end

    # function embed_catview_3()
    #     words = ["this", "is", "a", "test", "<unk>", "<num>"]
    #     d = Dict(word => Param(rand(8, 1)) for word in words)
        
    #     e1layer = Embed(8, d)(4, 1)
    #     c1layer = Conv((8,1,1), (1,1), (1,1), xavier, relu)(e1layer.t.o)
    #     f1layer = Dense(2, Knet.xavier, Knet.relu)(e1layer.t.o)

    #     function f(x)
    #         x = e1layer(x)
    #         x = c1layer(x)
    #         x = f1layer(x)
    #         return x
    #     end
        
    #     n = Network([e1layer, f1layer], f)
    #     @show n([["this", "is", "a", "test"]])

    #     dv = @diff n([["this", "is", "a", "test"]], [0x01])
    #     @show collect(params(dv))
    #     @test dv != nothing            
    # end

    @testset "catview" begin
        embed_catview_1()
        embed_catview_2()
        embed_catview_3()
        # embed_catview_2()
    end
end