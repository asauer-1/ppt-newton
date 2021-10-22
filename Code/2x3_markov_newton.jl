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

#Basis d=3, Gell-Mann matrices
s3 = Array{Complex{Float64}}(undef, 3, 3, 9)

s3[:,:,1] = [1 0 0; 0 1 0; 0 0 1]/sqrt(3)
s3[[1 2],[1 2], 2] = s2[:,:,2]
s3[[1 2],[1 2], 3] = s2[:,:,3]
s3[[1 2],[1 2], 4] = s2[:,:,4]
s3[[1 3],[1 3], 5] = s2[:,:,2]
s3[[1 3],[1 3], 6] = s2[:,:,3]
s3[[2 3],[2 3], 7] = s2[:,:,2]
s3[[2 3],[2 3], 8] = s2[:,:,3]
s3[:,:, 9] = [1 0 0; 0 1 0; 0 0 -2]/sqrt(6)


#Basis 2x3
k36 = Array{Complex{Float64}}(undef, 6, 6, 36)

for i in 0:3, j in 1:9
    k36[:,:,9*i+j] = kron(s2[:,:,i+1],s3[:,:,j])
end


#generate density matrix from vector a
#mul!(C,A,B,x,y)::  C := x*A*B + y*C
function mstate(a)
    state = Array{Complex{Float64}}(I/6,6,6)
    for i in 1:35
        mul!(state, a[i],  k36[:,:,i+1], 1, 1)
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

function newt5(s2,s3,s4,s5)
    return (1 - 10 * s2 + 20 * s3- 30 * s4 + 24*s5 + 15 * s2^2 - 20*s2*s3  >= 0 + 1e-16 )
end

function newt6(s2,s3,s4,s5,s6)
    return (1- 15 * s2 + 40 * s3 - 90 * s4 + 144*s5 - 120*s6
             + 45 * s2^2 - 120*s2*s3 + 90*s2*s4 + 40*s3^2 - 15*s2^3  >= 0 + 1e-16 )
end


#Check if vector a fulfills the Newton conditions
function checknewt(a)
    
    state = mstate(a)

    state = mstate(a)
    st2 = state^2
    st3 = state^3

    s2 = real(tr(st2))
    s3 = real(tr(st3))
    s4 = real(dot(st2',st2))
    s5 = real(dot(st2',st3))
    s6 = real(dot(st3',st3))

    return (newt2(s2) && newt3(s2,s3) && newt4(s2,s3,s4) && newt5(s2,s3,s4,s5) && newt6(s2,s3,s4,s5,s6))
end

#flip signs according to the partial transpose
function flip(a)
    return [(-1)^(18<=i<=26)*a[i] for i in 1:35]
end


#Hit and run algorithm for finding states
#find the next vector which is a state 
function check_loop(d, current)
    dir = random_vector(d)    

    vmax =  2 * sqrt(5 / 6)
    vmin = -2 * sqrt(5 / 6)

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
    d=35
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
    d=35
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

#test_run(10000)
inf_run(20000)
#inf_run(100000, "./results_2x3.txt")