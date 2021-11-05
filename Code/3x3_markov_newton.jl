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


#Basis 3x3
k81 = Array{Complex{Float64}}(undef, 9, 9, 81)

for i in 0:8, j in 1:9
    k81[:,:,9*i+j] = kron(s3[:,:,i+1],s3[:,:,j])
end

#generate density matrix from vector a
function mstate(a)
    state = Array{Complex{Float64}}(I/9,9,9)
    for i in 1:80
        mul!(state, a[i],  k81[:,:,i+1], 1, 1)
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

function newt7(s2,s3,s4,s5,s6,s7)
    return (1- 21 * s2 + 70 * s3 - 210 * s4 + 504*s5 - 840*s6 + 720*s7
             + 105 * s2^2 - 420*s2*s3 + 630*s2*s4 -504*s2*s5 + 280*s3^2 - 420*s3*s4 - 105*s2^3 +210*s2^2*s3 >= 0 + 1e-16 )
end
  
function newt8(s2,s3,s4,s5,s6,s7,s8)
    return (1- 28 * s2 + 112 * s3 - 420 * s4 + 1344*s5 - 3360*s6 + 5760*s7 - 5040*s8
             + 210 * s2^2 - 1120*s2*s3 + 2520*s2*s4 - 4032*s2*s5 + 3360*s2*s6 
             + 1120*s3^2  - 3360*s3*s4 + 2688*s3*s5 + 1260*s4^2 
             - 420*s2^3 + 1680*s2^2*s3 - 1260*s2^2*s4 - 1120*s2*s3^2 + 105*s2^4  >= 0 + 1e-16 )
end

function newt9(s2,s3,s4,s5,s6,s7,s8,s9)
    return (1- 36* s2 + 168 * s3 - 756 * s4 + 3024*s5 - 10080*s6 + 25920*s7 - 45360*s8 + 40320*s9
             + 378 * s2^2 - 2520*s2*s3 + 7560*s2*s4 - 18144*s2*s5 + 30240*s2*s6 - 25920*s2*s7
             + 3360*s3^2  - 15120*s3*s4 + 24192*s3*s5 - 20160*s3*s6 + 11340*s4^2 - 18144*s4*s5 
             - 1260*s2^3 + 7560*s2^2*s3 - 11340*s2^2*s4 + 9072*s2^2*s5 - 10080*s2*s3^2 + 15120*s2*s3*s4 + 2240*s3^3 
             + 945*s2^4 - 2520*s2^3*s3  >= 0 + 1e-16 )
end



#Check if vector a fulfills the Newton conditions
function checknewt(a)

    state = mstate(a)
    st2 = state^2
    st3 = state^3
    st4 = state^4

    s2 = real(tr(st2))
    s3 = real(tr(st3))
    s4 = real(tr(st4))

    # dot(a,b): a* . b
    # A': adjoint(A)    
    s5 = real(dot(st2',st3))
    s6 = real(dot(st3',st3))
    s7 = real(dot(st3',st4))
    s8 = real(dot(st4',st4))
    s9 = real(dot(st4',state^5))
    return (newt2(s2) && newt3(s2,s3) && newt4(s2,s3,s4) && newt5(s2,s3,s4,s5) && newt6(s2,s3,s4,s5,s6) 
    && newt7(s2,s3,s4,s5,s6,s7) && newt8(s2,s3,s4,s5,s6,s7,s8) && newt9(s2,s3,s4,s5,s6,s7,s8,s9) )
end


#flip signs according to the partial transpose
function flip(a)
    #gamma_2*B, gamma_5*B, gamma_7*B ^= 18:26, 45:53, 63:71  
    return [a[i]*(-1)^(i in [18:26; 45:53; 63:71]) for i in 1:80]
end


#Hit and run algorithm for finding states
#find the next vector which is a state 
function check_loop(d, current)
    dir = random_vector(d)    

    vmax =  2 * sqrt(8 / 9)
    vmin = -2 * sqrt(8 / 9)

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
    d=80
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
    d=80
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
inf_run(5000)
#inf_run(10000, "./results_3x3.txt")
