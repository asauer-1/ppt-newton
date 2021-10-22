using LinearAlgebra
using Random


function random_vector(n)
    v = randn(n)
    while norm(v) < .0001
        v = randn(n)   # random standard normal
    end
    return v / norm(v)  
end

#=
#Matrix definitions
Id = [1 0; 0 1]/sqrt(2)
sx = [0 1; 1 0]/sqrt(2)
sy = [0 -1im; 1im 0]/sqrt(2)
sz = [1 0; 0 -1]/sqrt(2)

L0 = [1 0 0;0 1 0; 0 0 1]/sqrt(3)
L1 = [0 1 0;1 0 0; 0 0 0]/sqrt(2)
L2 = [0 -1im 0; 1im 0 0; 0 0 0]/sqrt(2)
L3 = [1 0 0; 0 -1 0; 0 0 0]/sqrt(2)
L4 = [0 0 1; 0 0 0; 1 0 0]/sqrt(2)
L5 = [0 0 -1im; 0 0 0; 1im 0 0]/sqrt(2)
L6 = [0 0 0; 0 0 1; 0 1 0]/sqrt(2)
L7 = [0 0 0; 0 0 -1im; 0 1im 0]/sqrt(2)
L8 = [1 0 0; 0 1 0; 0 0 -2]/sqrt(6)


K0 = kron(Id, L0)
K1 = kron(Id, L1)
K2 = kron(Id, L2)
K3 = kron(Id, L3)
K4 = kron(Id, L4)
K5 = kron(Id, L5)
K6 = kron(Id, L6)
K7 = kron(Id, L7)
K8 = kron(Id, L8)
K9 = kron(sx, L0) 
K10 = kron(sx, L1)
K11 = kron(sx, L2) 
K12 = kron(sx, L3)
K13 = kron(sx, L4) 
K14 = kron(sx, L5) 
K15 = kron(sx, L6)
K16 = kron(sx, L7) 
K17 = kron(sx, L8)
K18 = kron(sy, L0) 
K19 = kron(sy, L1) 
K20 = kron(sy, L2)
K21 = kron(sy, L3) 
K22 = kron(sy, L4)
K23 = kron(sy, L5) 
K24 = kron(sy, L6) 
K25 = kron(sy, L7) 
K26 = kron(sy, L8) 
K27 = kron(sz, L0)
K28 = kron(sz, L1) 
K29 = kron(sz, L2)
K30 = kron(sz, L3) 
K31 = kron(sz, L4) 
K32 = kron(sz, L5) 
K33 = kron(sz, L6) 
K34 = kron(sz, L7)
K35 = kron(sz, L8)
=#
#=
function mstate(a)
     kron(Id, L0)/sqrt(6) + (a[1]*kron(Id, L1) + a[2]*kron(Id, L2) + a[3]*kron(Id, L3) + a[4]*kron(Id, L4) +
       a[5]*kron(Id, L5) + a[6]*kron(Id, L6) + a[7]*kron(Id, L7) + a[8]*kron(Id, L8) + a[9]*kron(sx, L0)  + 
       a[10]*kron(sx, L1) + a[11]*kron(sx, L2) + a[12]*kron(sx, L3) + a[13]*kron(sx, L4) + 
       a[14]*kron(sx, L5) + a[15]*kron(sx, L6) + a[16]*kron(sx, L7) + a[17]*kron(sx, L8) +  
       a[18]*kron(sy, L0) + a[19]*kron(sy, L1) + a[20]*kron(sy, L2) + a[21]*kron(sy, L3) +  
       a[22]*kron(sy, L4) + a[23]*kron(sy, L5) + a[24]*kron(sy, L6) + a[25]*kron(sy, L7) +  
       a[26]*kron(sy, L8) + a[27]*kron(sz, L0) + a[28]*kron(sz, L1) + a[29]*kron(sz, L2) +  
       a[30]*kron(sz, L3) + a[31]*kron(sz, L4) + a[32]*kron(sz, L5) + a[33]*kron(sz, L6) +  
       a[34]*kron(sz, L7) + a[35]*kron(sz, L8))
end

function mstate(a)
    K0/sqrt(6) + (a[1]*K1 + a[2]*K2 + a[3]*K3 + a[4]*K4 +
      a[5]*K5 + a[6]*K6 + a[7]*K7 + a[8]*K8 + a[9]*K9  + 
      a[10]*K10+ a[11]*K11 + a[12]*K12 + a[13]*K13 + 
      a[14]*K14 + a[15]*K15 + a[16]*K16 + a[17]*K17 +  
      a[18]*K18 + a[19]*K19 + a[20]*K20 + a[21]*K21 +  
      a[22]*K22 + a[23]*K23 + a[24]*K24 + a[25]*K25 +  
      a[26]*K26 + a[27]*K27 + a[28]*K28 + a[29]*K29 +  
      a[30]*K30 + a[31]*K31  + a[32]*K32 + a[33]*K33 +  
      a[34]*K34 + a[35]*K35)
end
=#


#full 2x3 system
function mstate_35(a)
    [1/sqrt(6)*a[27]+1/6*sqrt(3)*a[35]+1/6*sqrt(3)*a[8]+1/2*a[3]+1/2*a[30]+1/6  1/2*a[1]-1/2*im*a[2]+1/2*a[28]-1/2*im*a[29]  1/2*a[31]-1/2*im*a[32]+1/2*a[4]-1/2*im*a[5]  -1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]+1/6*sqrt(3)*a[17]-1/6*im*sqrt(3)*a[26]+1/2*a[12]-1/2*im*a[21]  1/2*a[10]-1/2*im*a[11]-1/2*im*a[19]-1/2*a[20]  1/2*a[13]-1/2*im*a[14]-1/2*im*a[22]-1/2*a[23];
    1/2*a[1]+1/2*im*a[2]+1/2*a[28]+1/2*im*a[29]  1/sqrt(6)*a[27]+1/6*sqrt(3)*a[35]+1/6*sqrt(3)*a[8]-1/2*a[3]-1/2*a[30]+1/6  1/2*a[33]-1/2*im*a[34]+1/2*a[6]-1/2*im*a[7]  1/2*a[10]+1/2*im*a[11]-1/2*im*a[19]+1/2*a[20]  -1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]+1/6*sqrt(3)*a[17]-1/6*im*sqrt(3)*a[26]-1/2*a[12]+1/2*im*a[21]  1/2*a[15]-1/2*im*a[16]-1/2*im*a[24]-1/2*a[25];
    1/2*a[31]+1/2*im*a[32]+1/2*a[4]+1/2*im*a[5]  1/2*a[33]+1/2*im*a[34]+1/2*a[6]+1/2*im*a[7]  1/sqrt(6)*a[27]-1/sqrt(3)*a[35]-1/sqrt(3)*a[8]+1/6  1/2*a[13]+1/2*im*a[14]-1/2*im*a[22]+1/2*a[23]  1/2*a[15]+1/2*im*a[16]-1/2*im*a[24]+1/2*a[25]  -1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]-1/sqrt(3)*a[17]+1/sqrt(3)*im*a[26];  
    1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]+1/6*sqrt(3)*a[17]+1/6*im*sqrt(3)*a[26]+1/2*a[12]+1/2*im*a[21]  1/2*a[10]-1/2*im*a[11]+1/2*im*a[19]+1/2*a[20]  1/2*a[13]-1/2*im*a[14]+1/2*im*a[22]+1/2*a[23]  -1/sqrt(6)*a[27]-1/6*sqrt(3)*a[35]+1/6*sqrt(3)*a[8]+1/2*a[3]-1/2*a[30]+1/6  1/2*a[1]-1/2*im*a[2]-1/2*a[28]+1/2*im*a[29]  -1/2*a[31]+1/2*im*a[32]+1/2*a[4]-1/2*im*a[5];
    1/2*a[10]+1/2*im*a[11]+1/2*im*a[19]-1/2*a[20]  1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]+1/6*sqrt(3)*a[17]+1/6*im*sqrt(3)*a[26]-1/2*a[12]-1/2*im*a[21]  1/2*a[15]-1/2*im*a[16]+1/2*im*a[24]+1/2*a[25]  1/2*a[1]+1/2*im*a[2]-1/2*a[28]-1/2*im*a[29]  -1/sqrt(6)*a[27]-1/6*sqrt(3)*a[35]+1/6*sqrt(3)*a[8]-1/2*a[3]+1/2*a[30]+1/6  -1/2*a[33]+1/2*im*a[34]+1/2*a[6]-1/2*im*a[7];
    1/2*a[13]+1/2*im*a[14]+1/2*im*a[22]-1/2*a[23]  1/2*a[15]+1/2*im*a[16]+1/2*im*a[24]-1/2*a[25]  1/sqrt(6)*im*a[18]+1/sqrt(6)*a[9]-1/sqrt(3)*a[17]-1/sqrt(3)*im*a[26]  -1/2*a[31]-1/2*im*a[32]+1/2*a[4]+1/2*im*a[5]  -1/2*a[33]-1/2*im*a[34]+1/2*a[6]+1/2*im*a[7]  -1/sqrt(6)*a[27]+1/sqrt(3)*a[35]-1/sqrt(3)*a[8]+1/6
    ]
end

function flip_35(a)
    return [(-1)^(18<=i<=26)*a[i] for i in 1:35]
end
function flip_b(a)
    return [(-1)^(i in [2,5,7,11,14,16,20,23,25,29,32,34])*a[i] for i in 1:35]
end


#8-dimensional subsystem
function mstate_8(a)
    kron(Id, L0)/sqrt(6) + a[1]*kron(sy, L1) + a[2]*kron(sy, L2) + a[3]*kron(sy, L3) +  
      a[4]*kron(sy, L4) + a[5]*kron(sy, L5) + a[6]*kron(sy, L6) + a[7]*kron(sy, L7) +  
      a[8]*kron(sy, L8)
end
function flip_8(a)
    return [-a[i] for i in 1:8]
end

#12-dimensional subsystem
function mstate_12(a)
#=
    kron(Id, L0)/sqrt(6) +
      a[1]*kron(sx, L1) + a[2]*kron(sx, L2) + a[3]*kron(sx, L3) + a[4]*kron(sx, L4) + 
      a[5]*kron(sy, L1) + a[6]*kron(sy, L2) + a[7]*kron(sy, L3) + a[8]*kron(sy, L4) +
      a[9]*kron(sz, L1) + a[10]*kron(sz, L2) + a[11]*kron(sz, L3) + a[12]*kron(sz, L4) 
=#
    [1/2*a[11]+1/6     -1/2*im*a[10]+1/2*a[9]     1/2*a[12]      1/2*a[3]-1/2*im*a[7]    1/2*a[1]-1/2*im*a[2]-1/2*im*a[5]-1/2*a[6]   1/2*a[4]-1/2*im*a[8];
     1/2*im*a[10]+1/2*a[9]    -1/2*a[11]+1/6     0    1/2*a[1]+1/2*im*a[2]-1/2*im*a[5]+1/2*a[6]    -1/2*a[3]+1/2*im*a[7]           0;
     1/2*a[12]               0              1/6              1/2*a[4]-1/2*im*a[8]                0                                     0;
     1/2*a[3]+1/2*im*a[7]   1/2*a[1]-1/2*im*a[2]+1/2*im*a[5]+1/2*a[6]    1/2*a[4]+1/2*im*a[8]   -1/2*a[11]+1/6   1/2*im*a[10]-1/2*a[9]   -1/2*a[12];
     1/2*a[1]+1/2*im*a[2]+1/2*im*a[5]-1/2*a[6]    -1/2*a[3]-1/2*im*a[7]        0        -1/2*im*a[10]-1/2*a[9]       1/2*a[11]+1/6         0;
     1/2*a[4]+1/2*im*a[8]          0      0               -1/2*a[12]              0                   1/6]

end

function flip_12(a)
    return [(-1)^(5<=i<=8)*a[i] for i in 1:12]
    #return [(-1)^(i in [2,6,10])*a[i] for i in 1:12]
end

#24-dimensional subsystem
function mstate_24(a)
    #=
    kron(Id, L0)/sqrt(6) + 
      a[1]*kron(sx, L1) + a[2]*kron(sx, L2) + a[3]*kron(sx, L3) + a[4]*kron(sx, L4) + 
      a[5]*kron(sx, L5) + a[6]*kron(sx, L6) + a[7]*kron(sx, L7) + a[8]*kron(sx, L8) +  
      a[9]*kron(sy, L1) + a[10]*kron(sy, L2) + a[11]*kron(sy, L3) + a[12]*kron(sy, L4) + 
      a[13]*kron(sy, L5) + a[14]*kron(sy, L6) + a[15]*kron(sy, L7) + a[16]*kron(sy, L8) + 
      a[17]*kron(sz, L1) + a[18]*kron(sz, L2) + a[19]*kron(sz, L3) + a[20]*kron(sz, L4) + 
      a[21]*kron(sz, L5) + a[22]*kron(sz, L6) + a[23]*kron(sz, L7) + a[24]*kron(sz, L8)
=#
      [1/6*sqrt(3)*a[24]+1/2*a[19]+1/6    1/2*a[17]-1/2*im*a[18]   1/2*a[20]-1/2*im*a[21]  -1/6*im*sqrt(3)*a[16]+1/6*sqrt(3)*a[8]-1/2*im*a[11]+1/2*a[3]   1/2*a[1]-1/2*a[10]-1/2*im*a[2]-1/2*im*a[9]   -1/2*im*a[12]-1/2*a[13]+1/2*a[4]-1/2*im*a[5];
       1/2*a[17]+1/2*im*a[18]   1/6*sqrt(3)*a[24]-1/2*a[19]+1/6    1/2*a[22]-1/2*im*a[23]   1/2*a[1]+1/2*a[10]+1/2*im*a[2]-1/2*im*a[9]   -1/6*im*sqrt(3)*a[16]+1/6*sqrt(3)*a[8]+1/2*im*a[11]-1/2*a[3]      -1/2*im*a[14]-1/2*a[15]+1/2*a[6]-1/2*im*a[7]; 
       1/2*a[20]+1/2*im*a[21]   1/2*a[22]+1/2*im*a[23]      -1/3*sqrt(3)*a[24]+1/6         -1/2*im*a[12]+1/2*a[13]+1/2*a[4]+1/2*im*a[5]   -1/2*im*a[14]+1/2*a[15]+1/2*a[6]+1/2*im*a[7]    1/3*im*sqrt(3)*a[16]-1/3*sqrt(3)*a[8];
       1/6*im*sqrt(3)*a[16]+1/6*sqrt(3)*a[8]+1/2*im*a[11]+1/2*a[3]    1/2*a[1]+1/2*a[10]-1/2*im*a[2]+1/2*im*a[9]   1/2*im*a[12]+1/2*a[13]+1/2*a[4]-1/2*im*a[5]  -1/6*sqrt(3)*a[24]-1/2*a[19]+1/6    -1/2*a[17]+1/2*im*a[18]   -1/2*a[20]+1/2*im*a[21];
       1/2*a[1]-1/2*a[10]+1/2*im*a[2]+1/2*im*a[9]   1/6*im*sqrt(3)*a[16]+1/6*sqrt(3)*a[8]-1/2*im*a[11]-1/2*a[3]    1/2*im*a[14]+1/2*a[15]+1/2*a[6]-1/2*im*a[7]  -1/2*a[17]-1/2*im*a[18]     -1/6*sqrt(3)*a[24]+1/2*a[19]+1/6  -1/2*a[22]+1/2*im*a[23];
       1/2*im*a[12]-1/2*a[13]+1/2*a[4]+1/2*im*a[5]  1/2*im*a[14]-1/2*a[15]+1/2*a[6]+1/2*im*a[7]  -1/3*im*sqrt(3)*a[16]-1/3*sqrt(3)*a[8]  -1/2*a[20]-1/2*im*a[21]   -1/2*a[22]-1/2*im*a[23]   1/3*sqrt(3)*a[24]+1/6    ]     

end
function flip_24(a)
    return [(-1)^(9<=i<=16)*a[i] for i in 1:24]
    #return [(-1)^(i in [2,5,7,10,13,15,18,21,23])*a[i] for i in 1:24]
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


function flip(type, a)
    if (type == 35)
        return flip_35(a)
    elseif (type == 24)
        return flip_24(a)
    elseif (type == 12)
        return flip_12(a)
    elseif (type == 8)
        return flip_8(a)
    else
        println("Wrong Type")
    end
end

function mstate(type, a)
    if (type == 35)
        return mstate_35(a)
    elseif (type == 24)
        return mstate_24(a)
    elseif (type == 12)
        return mstate_12(a)
    elseif (type == 8)
        return mstate_8(a)
    else
        println("Wrong Type")
    end
end


function checknewt(type,a)
    
    state = mstate(type, a)

    st2 = state^2
    st3 = state^3

    # dot(a,b): a* . b
    # A': adjoint(A)    
    s2 = real(tr(st2))
    s3 = real(tr(st3))
    s4 = real(dot(st2',st2))
    s5 = real(dot(st2',st3))
    s6 = real(dot(st3',st3))

    return (newt2(s2) && newt3(s2,s3) && newt4(s2,s3,s4) && newt5(s2,s3,s4,s5) && newt6(s2,s3,s4,s5,s6))
end

function check_loop(d, current)
    dir = random_vector(d)

    vmax = 2 * sqrt(5/6)
    vmin = -2 *sqrt(5/6)
    
    v = vmin + (vmax-vmin)*rand()
    a = current + v * dir   
        
    while !checknewt(d, a)
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax-vmin)*rand()
        a = current + v * dir    
    end

    #returns the new vector and if it fulfills PPT   
    return a, checknewt(d, flip(d, a)) 
end



function test_run(type=35, num=100000)

results = [0,0]
current = zeros(type)

for _ in 1:10

    ppt = 0 
    @time for _ in 1:num
        current, pc = check_loop(type, current)
        ppt += pc
    end

    results[1] += num
    results[2] += ppt

    str = string(results[1], ";  ", results[2]/results[1], ";  ", ppt)
    println(str)
end

end

#finds new vectors and prints the results to the specified file after num steps, runs until interrupted
function inf_run(type=35, num=100000, write_to="")
    results = [0,0]
    current = zeros(type)
    
    while true
    
        ppt = 0 
        for _ in 1:num
            current, pc = check_loop(type, current)
            ppt += pc
        end
    
        results[1] += num
        results[2] += ppt
    
        str = string(results[1], ";  ", results[2]/results[1], ";  ", ppt)
        println(str)
        
        if !isempty(write_to)
            io = open(string(write_to), "a");
            #io = open(string("results_", type, ".txt"), "a");
            write(io, str, "\n");
            close(io)
        end
    end
end




#Start the computation with the desired parameters

test_run(35,100000)
#inf_run(35,100000,"results_35dim.txt")
