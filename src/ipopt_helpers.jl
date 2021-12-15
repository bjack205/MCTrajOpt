function parseipoptoutput(file)
    io = open(file, "r")

    isheaderline(line) = !isempty(line) && split(line)[1] == "iter"
    isdataline(line) = !isempty(line) && !isnothing(tryparse(Float64, split(line)[1]))

    # Find the first log header line
    # Assumes it starts with "iter"
    line = readline(io)
    while !isheaderline(line) 
        line = readline(io)
    end

    # Create data dictionary, with one entry per header
    headers = split(line)
    data = Dict(header => Float64[] for header in headers)

    # Loop through the file
    while !eof(io)
        line = readline(io)
        if isdataline(line)
            i = 1
            for entry in split(line)
                res = tryparse(Float64, entry)
                if !isnothing(res)
                    push!(data[headers[i]], res)
                    i += 1
                end
            end
        end
    end
    return data
end