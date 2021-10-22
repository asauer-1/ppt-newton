using LinearAlgebra
using Random


function random_vector(n)
    v= randn(n) # random standard normal
    while norm(v) < .0001
        v = randn(n) 
    end
    return v / norm(v)  
end


#Basis d=2
s2 = Array{Complex{Float64}}(undef, 2, 2, 4)

s2[:,:,1] = [1 0; 0 1]/sqrt(2)
s2[:,:,2] = [0 1; 1 0]/sqrt(2)
s2[:,:,3] = [0 -1im; 1im 0]/sqrt(2)
s2[:,:,4] = [1 0; 0 -1]/sqrt(2)

#Basis 2x2
k16 = Array{Complex{Float64}}(undef, 4, 4, 16)
for i in 0:3, j in 1:4
    k16[:,:,i*4+j] =  kron(s2[:,:,i+1],s2[:,:,j])
end

#generate density matrix from vector a
#mul!(C,A,B,x,y)::  C := x*A*B + y*C
function mstate(a)
    state = Array{Complex{Float64}}(I/4,4,4)
    for i in 1:15
        mul!(state, a[i],  k16[:,:,i+1], 1, 1)
    end
    return state
end


#Newton
function newt2(s2) 
    return (1 - s2 >= 0 + 1e-16 )
end

function newt3(s2,s3)
    return (1 - 3 * s2 + 2 * s3 >= 0 + 1e-16  ) 
end

function newt4(s2,s3,s4)
    return (1 - 6 * s2 + 8 * s3 + 3 * s2^2 - 6 * s4 >= 0 + 1e-16 )
end


#Check if vector a fulfills the Newton conditions
function checknewt(a)
    
    state = mstate(a)

    s2 = real(tr(state^2))
    s3 = real(tr(state^3))
    s4 = real(tr(state^4))
    return (newt2(s2) && newt3(s2,s3) && newt4(s2,s3,s4)) 
end

#flip signs according to the partial transpose
function flip(a)
    return [a[i]*(-1)^(i in 8:11) for i in 1:15]
end


#Hit and run algorithm for finding states

#=
#old version, slightly slower
function getMax(start, dir, pow)
    v = 2* sqrt(3 / 4)
    for i in 0:pow
        a = start + v * dir 
        checknewt(a) ? (v += 2.0^(-i)) : (v -= 2.0^(-i)) 
    end
    return v
end
function check_loop(d, current)
    dir = random_vector(d)
    vmax = getMax(current, dir, 5) + 2^-6
    vmin = -getMax(current, -dir, 5) -2^-6
    @label failed
    v = vmin+ (vmax-vmin)* rand()
    a = current + v * dir
    if  checknewt(a) 
        current = a
    else
        @goto failed
    end  
    return current, checknewt(flip(current)) 
end
=#

#find the next vector which is a state 
function check_loop(d, current)
    dir = random_vector(d)    

    vmax =  2 * sqrt(3 / 4)
    vmin = -2 * sqrt(3 / 4)

    v = vmin + (vmax-vmin)*rand()
    a = current + v * dir

    while !checknewt(a)
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = current + v * dir    
    end

    #returns the new vector and if it fulfills PPT
    return a, checknewt(flip(a)) 
end


#times the creation and check of n vectors
function  test_run(n)
    d=15
    results = 0
    current = zeros(d)
    @time for _ in 1:n 
        current, ppt = check_loop(d,current)
        results += ppt
    end
    println(results/n)
end 


#finds new vectors and prints the results to the specified file after num steps, runs until interrupted
function inf_run(num=100000, write_to="")
    d=15
    results = [0,0]

    current = zeros(d)
    
    while true
        for _ in 1:num 
            current, ppt = check_loop(d,current)
            results[2] += ppt
        end
        results[1] += num
        
        str = string(results[1], ";  ", results[2]/results[1])
        println(str)
        
        if !isempty(write_to)
            io = open(string(write_to), "a");
            write(io, str, "\n");
            close(io)
        end
    end
end


#Start the computations with the desired parameters

test_run(10000)
#inf_run(20000)
#inf_run(100000, "./results_2x2.txt")
