
function dataloader(data::Matrix)
    data = sortslices(data, dims=1);
    # ! SORTSILCES IS TYPE UNSTABLE
    
    if size(data,2) == 2
        return OneVarPolydata{eltype(data)}(data[:,1],data[:,2])
    elseif size(data,2) == 3
        x1,x2 = unique( data[:,1] ), unique( data[:,2] );
        return TwoVarPolydata{eltype(data)}(x1,x2,data[:,3])
    else
        uniques = [unique(data[:,k]) for k in 1:size(data,2)-1] # retrieve unique vectors in data
        NVarPolydata{eltype(data), size(data,2)-1 }(uniques,data[:,end])
    end

end

function dataloader(values::Vector,x::Vector...)
    # ! assumes data was previously sorted
    # ? some kind of data = hcat(x...,values) and call the previous method

    if length(x) == 1
        return OneVarPolydata{eltype(values)}(x[1],values)
    elseif length(x) == 2
        x1,x2 = unique( x[1] ), unique( x[2] ) ;
        return TwoVarPolydata{eltype(values)}(x1,x2,values)
    else
        uniques = [unique(x[k]) for k in 1:length(x)] # retrieve unique vectors in data
        NVarPolydata{eltype(values), length(x) }(uniques,values)
    end
end

