#=

Copyright (c) 2019 Ali Mert Ceylan

=#


using Pkg
using WordTokenizers

sstbddir = joinpath(@__DIR__, "sstb")

labels = Dict("very neg"=>0x01, "neg"=>0x02, "neu"=>0x03, "pos"=>0x04, "very pos"=>0x05)

function sstbd_finegrained(file_path)
	sstbd_lines = nothing
	open(file_path, "r") do sstbd_file
	    sstbd_lines = readlines(sstbd_file)
	end


	sstbd_lines_processed = split.(sstbd_lines, "<->")
	_sstbd_x, _sstbd_y = [tokenize(sstbdline[1]) for sstbdline in sstbd_lines_processed], [labels[sstbdline[2]] for sstbdline in sstbd_lines_processed]


	return _sstbd_x, _sstbd_y
end

function sstbd_binary(file_path)
	sstbd_lines = nothing
	open(file_path, "r") do sstbd_file
	    sstbd_lines = readlines(sstbd_file)
	end


	sstbd_lines_processed = split.(sstbd_lines, "<->")
	_sstbd_x = Array{Any, 1}()
	_sstbd_y = Array{UInt8, 1}()

	for sstbdline in sstbd_lines_processed
		if(labels[sstbdline[2]] != 0x03)
			push!(_sstbd_x, tokenize(sstbdline[1]))
			push!(_sstbd_y, labels[sstbdline[2]]>0x03 ? 0x01 : 0x02)
		end
	end

	return _sstbd_x, _sstbd_y
end
