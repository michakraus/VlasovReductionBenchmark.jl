# The exception that `f()` throws, or `nothing` when it throws none.
function thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end
