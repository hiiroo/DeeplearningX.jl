#=

Copyright (c) 2019 Ali Mert Ceylan

=#


using Pkg

wikicorpusdir = joinpath(@__DIR__, "wikicorpus")

function wikicorpus(file_path)
	global _wikicorpus_x, _wikicorpus_y

	wcf_lines = nothing
	open(file_path, "r") do wcf_file
	    wcf_lines = readlines(wcf_file)
	end

	if !@isdefined(_wikicorpus_x)
		wcf_lines_processed = split.(wcf_lines, " ")
		_wikicorpus_x, _wikicorpus_y = [wcfline[1:end-1] for wcfline in wcf_lines_processed], [parse(UInt8, wcfline[end])+0x01 for wcfline in wcf_lines_processed]

		# _wikicorpus_xtrn, _wikicorpus_ytrn = wcf_x[train_inst[1]:train_inst[2]], wcf_y[train_inst[1]:train_inst[2]]
		# _wikicorpus_xtst, _wikicorpus_ytst = wcf_x[test_inst[1]:test_inst[2]], wcf_y[test_inst[1]:test_inst[2]]
	end

	return _wikicorpus_x, _wikicorpus_y
end

function wikicorpusdata(file_path;batch_size=100)

	if !@isdefined(_wikicorpus_xtrn); wikicorpus(file_path); end


	wcf_train = Knet.minibatch(_wikicorpus_xtrn, _wikicorpus_ytrn, batch_size, shuffle=true)
	wcf_test = Knet.minibatch(_wikicorpus_xtst, _wikicorpus_ytst, batch_size, shuffle=true)

	return wcf_train, wcf_test
end