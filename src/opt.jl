#=

Copyright (c) 2019 Ali Mert Ceylan

=#


mutable struct SGD 
    learning_rate
end


mutable struct ADAM
    learning_rate
    alpha
    beta1
    beta2
    mt
    vt
    t
end


function iterate(model, data, opt::SGD)
	batch_x, batch_y = data

    dv = @diff model(batch_x, batch_y)

    dparams = @parameters dv

    for pidx in 1:length(dparams[1:end-1])
        dw = grad(dv, dparams[pidx])
        dw .-=opt.learning_rate*dw
    end
end


function iterate(model, data, opt::ADAM)
    batch_x, batch_y = data
        
    dv = @diff model(batch_x, batch_y)
    
    dparams = @parameters dv
    
    for pidx in 1:length(dparams[1:end-1])
        dw = grad(dv, dparams[pidx])
        mt = opt.beta1*opt.mt.+(1-opt.beta1).*dw
        vt = opt.beta2*vt.+(1-opt.beta2).*(dw.^2)
        mt = mt./(1-opt.beta1^opt.t)
        vt = vt./(1-opt.beta2^opt.t)
        dparams[pidx] .-=opt.alpha.*(opt.mt./(sqrt.(opt.vt).+opt.learning_rate))
    end
    
end
