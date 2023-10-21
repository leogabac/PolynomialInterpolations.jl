
function data2()
    x = collect(Float64, range(0, stop = 2π, length = 10)); 
    y = round.(sin.(x), digits=2); 
    return  [x y]
end

function data3()
    x = collect(Float64, range(0, stop = 2π, length = 10));
    y = collect(Float64, range(0, stop = 2π, length = 10));
    m = length(x); n = length(y);
    c = 1;
    data = Array{Float64}(undef, m * n, 3);
    for i in 1:m
        for j in 1:n
            data[c,:] = [ x[i], y[j], round(sin(x[i]) + cos(y[j]), digits=2)]';
            c += 1;
        end 
    end
    return data
end

function datan()
    x = collect(Float64, 1:4);
    y = collect(Float64, 1:4);
    z = collect(Float64, -1:3);
    t = collect(Float64, 0:3);

    m = length(x); n = length(y); p = length(z); q = length(t);
    data = Array{Float64}(undef, m * n * p * q, 5);
    dx  = 2.5;
    dy = -0.4;
    c = 1; 
    for i in 1:m
        for j in 1:n
            for k in 1:p
                for l in 1:q
                    data[c,:] = [x[i], y[j], z[k], t[l], round(sin(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
                    c += 1;
                end
            end
        end 
    end
    return data
end