#=

Copyright (c) 2019 Ali Mert Ceylan

=#


using Pkg
using WordTokenizers

trecqcddir = joinpath(@__DIR__, "trecqc")

labels = Dict("ABBR"=>0x01, "ENTY"=>0x02, "DESC"=>0x03, "HUM"=>0x04, "LOC"=>0x05, "NUM"=>0x06)

function trecqc(file_path)
	trecqc_lines = nothing
	open(file_path, "r") do trecqc_file
	    trecqc_lines = readlines(trecqc_file)
	end


	trecqc_lines_processed = split.(trecqc_lines, ":")
	_trecqc_x, _trecqc_y = [tokenize(trecqcline[2]) for trecqcline in trecqc_lines_processed], [labels[trecqcline[1]] for trecqcline in trecqc_lines_processed]


	return _trecqc_x, _trecqc_y
end
