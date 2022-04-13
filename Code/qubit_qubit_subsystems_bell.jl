using LinearAlgebra
using Random

function random_vector(n)
    v= randn(n)
    while norm(v) < .0001
        v = randn(n) # random standard normal
    end
    return v / norm(v)  
end

#Conditions on the vectors can be determined without computing the density matrix

#Bell diagonal
function belln2(a) 
    return 4*(a[1]^2 + a[2]^2 + a[3]^2) <= 3 +1e-16
end

function belln3(a) 
    return 4*(a[1]^2 + a[2]^2 + 4*a[1]*a[2]*a[3] + a[3]^2) <= 1 +1e-16
end

function belln4(a) 
    return 16*a[1]^4 + 16*a[2]^4 + (1 - 4*a[3]^2)^2 >= 8*(8*a[1]*a[2]*a[3] + a[2]^2*(1 + 4*a[3]^2) + a[1]^2*(1 + 4*a[2]^2 + 4*a[3]^2)) -1e-16
end

function flip_b(a)
    return [a[1], -a[2], a[3]]
end


#x state
function xn2(a) 
    return 4*(a[1]^2 + a[2]^2 + a[3]^2 + a[4]^2 + a[5]^2 + a[6]^2 + a[7]^2) <= 3 + 1e-16
end

function xn3(a) 
    return 1e-16 + 1 + 16 * a[1] * (a[2] * a[5] + a[3] * a[6] - a[4] * a[7]) >= 
    4*(a[1]^2 + a[2]^2 + a[3]^2 + a[4]^2 + a[5]^2 + a[6]^2 + a[7]^2)
end

function xn4(a) 
    return 1e-16 + 1 + 16*a[1]^4 + 16* a[2]^4 + 16* a[3]^4 + 32 *a[3]^2 *a[4]^2 + 16 *a[4]^4 +
    32* a[3]^2 * a[5]^2 + 32* a[4]^2 *a[5]^2 + 16* a[5]^4 + 32* a[4]^2* a[6]^2 + 
    32* a[5]^2* a[6]^2 + 16* a[6]^4 + 32* a[3]^2* a[7]^2 + 32* a[5]^2 *a[7]^2 + 
    32* a[6]^2* a[7]^2 + 16* a[7]^4 + 64* a[1] *(a[2]* a[5] + a[3]* a[6] - a[4]* a[7]) + 
    8* a[2]^2 *(-1 + 4 *a[3]^2 + 4* a[4]^2 - 4* a[5]^2 + 4* a[6]^2 + 4* a[7]^2) >= 
    8 *(a[4]^2 + a[5]^2 + a[6]^2 + a[3]^2 *(1 + 4* a[6]^2) - 16* a[2] *a[4] *a[5] *a[7] + a[7]^2 + 
    4* a[4]^2 *a[7]^2 + 16* a[3]* a[6] *(a[2] *a[5] - a[4] *a[7]) + 
    a[1]^2 *(1 + 4* a[2]^2 + 4* a[3]^2 + 4* a[4]^2 + 4* a[5]^2 + 4* a[6]^2 + 4 *a[7]^2))
end

function flip_x(a)
    [a[1], -a[2], a[3], -a[4], a[5], a[6], a[7]]
end


#rebit
function newt2(a) 
    return (-4*a[1]^2 - 4*a[2]^2 - 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 - 4*a[6]^2 - 4*a[7]^2 - 4*a[8]^2 - 4*a[9]^2 + 3 >= 0 + 1e-16 )
end

function newt3(a)
    return  (16*a[1]*a[3]*a[5] + 16*a[2]*a[3]*a[6] - 4*a[1]^2 - 4*a[2]^2 - 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 - 4*a[6]^2 - 4*a[7]^2 
    + 16*(a[1]*a[4] + a[6]*a[7])*a[8] - 4*a[8]^2 + 16*(a[2]*a[4] - a[5]*a[7])*a[9] - 4*a[9]^2 + 1 >= 0 + 1e-16 )
end

function newt4(a)
    return (16*a[1]^4 + 16*a[2]^4 + 16*a[3]^4 + 16*a[4]^4 + 16*a[5]^4 + 16*a[6]^4 + 16*a[7]^4 + 16*a[8]^4 + 16*a[9]^4 
    + 8*(4*a[1]^2 - 1)*a[2]^2 - 8*(4*a[1]^2 + 4*a[2]^2 + 1)*a[3]^2 - 8*(4*a[1]^2 + 4*a[2]^2 - 4*a[3]^2 + 1)*a[4]^2 
    + 64*a[1]*a[3]*a[5] - 8*(4*a[1]^2 - 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 + 1)*a[5]^2 + 8*(4*a[1]^2 - 4*a[2]^2 - 4*a[3]^2 + 4*a[4]^2 + 4*a[5]^2 - 1)*a[6]^2 
    + 8*(4*a[1]^2 + 4*a[2]^2 + 4*a[3]^2 + 4*a[4]^2 - 4*a[5]^2 - 4*a[6]^2 - 1)*a[7]^2 - 8*(4*a[1]^2 - 4*a[2]^2 - 4*a[3]^2 + 4*a[4]^2 - 4*a[5]^2 
    + 4*a[6]^2 + 4*a[7]^2 + 1)*a[8]^2 + 8*(4*a[1]^2 - 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 + 4*a[6]^2 - 4*a[7]^2 + 4*a[8]^2 - 1)*a[9]^2 
    - 8*a[1]^2 - 64*(2*a[1]*a[2]*a[5] - a[2]*a[3])*a[6] + 128*(a[2]*a[4]*a[5] - a[1]*a[4]*a[6])*a[7] - 64*(2*a[3]*a[4]*a[5] - a[1]*a[4] 
    + (2*a[2]*a[3] - a[6])*a[7])*a[8] - 64*(2*a[3]*a[4]*a[6] - a[2]*a[4] - (2*a[1]*a[3] - a[5])*a[7] + 2*(a[1]*a[2] - a[5]*a[6])*a[8])*a[9] + 1 >= 0 + 1e-16 )
end

function flip_r(a)
    return [a[i]*(-1)^(i in [7]) for i in 1:9 ]
end


#full state
function newr2(a) 
    3/4 - a[1]^2 - a[10]^2 - a[11]^2 - a[12]^2 - a[13]^2 - a[14]^2 - a[15]^2 - 
    a[2]^2 - a[3]^2 - a[4]^2 - a[5]^2 - a[6]^2 - a[7]^2 - a[8]^2 - a[9]^2 >= 0 + 1e-16
end
    
function newr3(a)
    return (-4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 - 4*a[13]^2 - 4*a[14]^2 - 4*a[15]^2 - 4*a[2]^2 
    - 4*a[3]^2 + 16*(a[10]*a[2] + a[13]*a[3])*a[4] - 4*a[4]^2 + 16*(a[11]*a[2] + a[14]*a[3])*a[5] - 4*a[5]^2 
    + 16*(a[12]*a[2] + a[15]*a[3])*a[6] - 4*a[6]^2 + 16*(a[12]*a[14] - a[11]*a[15] + a[1]*a[4])*a[7] 
    - 4*a[7]^2 - 16*(a[12]*a[13] - a[10]*a[15] - a[1]*a[5])*a[8] - 4*a[8]^2 + 16*(a[11]*a[13] - a[10]*a[14] 
    + a[1]*a[6])*a[9] - 4*a[9]^2 + 1 >= 0 + 1e-16 )

end
    
function newr4(a)
    return (16*a[1]^4 + 16*a[10]^4 + 16*a[11]^4 + 16*a[12]^4 + 16*a[13]^4 + 128*a[10]*a[11]*a[13]*a[14] 
    + 16*a[14]^4 + 16*a[15]^4 + 16*a[2]^4 + 16*a[3]^4 + 16*a[4]^4 + 16*a[5]^4 + 16*a[6]^4 + 16*a[7]^4
    + 16*a[8]^4 + 16*a[9]^4 + 8*(4*a[1]^2 - 1)*a[10]^2 + 8*(4*a[1]^2 + 4*a[10]^2 - 1)*a[11]^2 
    + 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 - 1)*a[12]^2 + 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 
    - 4*a[12]^2 - 1)*a[13]^2 + 8*(4*a[1]^2 - 4*a[10]^2 + 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 - 1)*a[14]^2 
    + 8*(4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 - 1)*a[15]^2 + 8*(4*a[1]^2 
    - 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 + 4*a[15]^2 - 1)*a[2]^2 - 128*(a[10]*a[13]
    + a[11]*a[14] + a[12]*a[15])*a[2]*a[3] + 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 + 4*a[12]^2 - 4*a[13]^2
    - 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 - 1)*a[3]^2 - 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 
    + 4*a[13]^2 - 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 + 4*a[3]^2 + 1)*a[4]^2 - 8*(4*a[1]^2 - 4*a[10]^2
    + 4*a[11]^2 - 4*a[12]^2 - 4*a[13]^2 + 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 
    + 1)*a[5]^2 - 8*(4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 - 4*a[13]^2 - 4*a[14]^2 + 4*a[15]^2 
    + 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 + 1)*a[6]^2 - 8*(4*a[1]^2 - 4*a[10]^2 + 4*a[11]^2 
    + 4*a[12]^2 - 4*a[13]^2 + 4*a[14]^2 + 4*a[15]^2 - 4*a[2]^2 - 4*a[3]^2 + 4*a[4]^2 - 4*a[5]^2 
    - 4*a[6]^2 + 1)*a[7]^2 - 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 + 4*a[13]^2 - 4*a[14]^2 
    + 4*a[15]^2 - 4*a[2]^2 - 4*a[3]^2 - 4*a[4]^2 + 4*a[5]^2 - 4*a[6]^2 - 4*a[7]^2 + 1)*a[8]^2 
    - 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 - 4*a[15]^2 - 4*a[2]^2 
    - 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 + 4*a[6]^2 - 4*a[7]^2 - 4*a[8]^2 + 1)*a[9]^2 - 8*a[1]^2 
    + 128*(a[10]*a[12]*a[13] + a[11]*a[12]*a[14])*a[15] - 64*(2*a[1]*a[12]*a[14] - 2*a[1]*a[11]*a[15] 
    - a[10]*a[2] - a[13]*a[3])*a[4] + 64*(2*a[1]*a[12]*a[13] - 2*a[1]*a[10]*a[15] + a[11]*a[2] 
    + a[14]*a[3] - 2*(a[10]*a[11] + a[13]*a[14])*a[4])*a[5] - 64*(2*a[1]*a[11]*a[13] - 2*a[1]*a[10]*a[14] 
    - a[12]*a[2] - a[15]*a[3] + 2*(a[10]*a[12] + a[13]*a[15])*a[4] + 2*(a[11]*a[12] 
    + a[14]*a[15])*a[5])*a[6] - 64*(2*a[1]*a[10]*a[2] + 2*a[1]*a[13]*a[3] - a[12]*a[14] + a[11]*a[15] 
    - a[1]*a[4] - 2*(a[15]*a[2] - a[12]*a[3])*a[5] + 2*(a[14]*a[2] - a[11]*a[3])*a[6])*a[7] 
    - 64*(2*a[1]*a[11]*a[2] + 2*a[1]*a[14]*a[3] + a[12]*a[13] - a[10]*a[15] + 2*(a[15]*a[2] 
    - a[12]*a[3])*a[4] - a[1]*a[5] - 2*(a[13]*a[2] - a[10]*a[3])*a[6] - 2*(a[10]*a[11] + a[13]*a[14] 
    - a[4]*a[5])*a[7])*a[8] - 64*(2*a[1]*a[12]*a[2] + 2*a[1]*a[15]*a[3] - a[11]*a[13] + a[10]*a[14] 
    - 2*(a[14]*a[2] - a[11]*a[3])*a[4] + 2*(a[13]*a[2] - a[10]*a[3])*a[5] - a[1]*a[6] - 2*(a[10]*a[12] 
    + a[13]*a[15] - a[4]*a[6])*a[7] - 2*(a[11]*a[12] + a[14]*a[15] - a[5]*a[6])*a[8])*a[9] + 1  >= 0+1e-16 )

end

function flip_f(a)
    return [a[i]*(-1)^(i in [5,8,11,14]) for i in 1:15 ]
end



#transforms measurement vectors a1,a2,b1,b2 to c, with Bell inequality c*x<=b; x:=vector representing the state 
function vec_prod_bell(type, a1,a2,b1,b2)
    if type == 3
        return [2*(a1[i]*b1[i]+a2[i]*b1[i]+a1[i]*b2[i]-a2[i]*b2[i])   for i in 1:3]

    elseif type == 7
        return 2* [a1[3]*b1[3] + a2[3]*b1[3]  + a1[3]*b2[3] - a2[3]*b2[3] ,
        a1[2]*b1[1] + a2[2]*b1[1] + a1[2]*b2[1] - a2[2]*b2[1],   
        a1[2]*b1[2] + a2[2]*b1[2] + a1[2]*b2[2] - a2[2]*b2[2],
        a1[1]*b1[2] + a2[1]*b1[2] + a1[1]*b2[2] - a2[1]*b2[2], 
        a1[1]*b1[1] + a2[1]*b1[1] + a1[1]*b2[1] - a2[1]*b2[1]]

    elseif type == 9
        return 2* [a1[1]*b1[1] + a2[1]*b1[1]  + a1[1]*b2[1] - a2[1]*b2[1] ,
        a1[1]*b1[3] + a2[1]*b1[3] + a1[1]*b2[3] - a2[1]*b2[3],   
        a1[2]*b1[2] + a2[2]*b1[2] + a1[2]*b2[2] - a2[2]*b2[2],
        a1[3]*b1[1] + a2[3]*b1[1] + a1[3]*b2[1] - a2[3]*b2[1], 
        a1[3]*b1[3] + a2[3]*b1[3] + a1[3]*b2[3] - a2[3]*b2[3]]

    elseif type == 15
        return  2 * [a1[1]*b1[1] + a2[1]*b1[1] + a1[1]*b2[1] - a2[1]*b2[1],
       a1[1]*b1[2] + a2[1]*b1[2] + a1[1]*b2[2] - a2[1]*b2[2],
       a1[1]*b1[3] + a2[1]*b1[3] + a1[1]*b2[3] - a2[1]*b2[3],
       a1[2]*b1[1] + a2[2]*b1[1] + a1[2]*b2[1] - a2[2]*b2[1],
       a1[2]*b1[2] + a2[2]*b1[2] + a1[2]*b2[2] - a2[2]*b2[2],
       a1[2]*b1[3] + a2[2]*b1[3] + a1[2]*b2[3] - a2[2]*b2[3],
       a1[3]*b1[1] + a2[3]*b1[1] + a1[3]*b2[1] - a2[3]*b2[1],
       a1[3]*b1[2] + a2[3]*b1[2] + a1[3]*b2[2] - a2[3]*b2[2],
       a1[3]*b1[3] + a2[3]*b1[3] + a1[3]*b2[3] - a2[3]*b2[3]] 
    else
        println("Wrong type!")
    end
end


function flip(type, a)
    if type == 3
        return flip_b(a)
    elseif type == 7
        return flip_x(a)    
    elseif type == 9
        return flip_r(a)
    elseif type == 15
        return flip_f(a)
    else
        println("Wrong type!")
    end
end 

function checknewt(type, a)
    if type == 3
        return (belln2(a) && belln3(a) && belln4(a))
    elseif type == 7
        return (xn2(a) && xn3(a) && xn4(a))
    elseif type == 9
        return (newt2(a) && newt3(a) && newt4(a))
    elseif type == 15
        return (newr2(a) && newr3(a) && newr4(a))
    else
        println("Wrong type!")
    end
end 


# generates the 12 Bell measurements 
function bell_meas(type)
    if type == 3
        dim = 3
    elseif type == 7 || type == 9
        dim = 5
    elseif type == 15
        dim = 9
    else
        println("Wrong type!")
    end

    m_a = [1 0 0; 0 1 0; 0 0 1]
    m_b = [1 1 0; 1 -1 0; 1 0 1; 1 0 -1 ; 0 1 1 ; 0 1 -1]/sqrt(2)
    vec_bell = zeros(12, dim)

    for i in 1:4
        vec_bell[0+i,:] = vec_prod_bell(type,(-1)^i*m_a[1,:],(-1)^(i÷2)*m_a[2,:],m_b[1,:],m_b[2,:])
        vec_bell[4+i,:] = vec_prod_bell(type,(-1)^i*m_a[1,:],(-1)^(i÷2)*m_a[3,:],m_b[3,:],m_b[4,:])
        vec_bell[8+i,:] = vec_prod_bell(type,(-1)^i*m_a[2,:],(-1)^(i÷2)*m_a[3,:],m_b[5,:],m_b[6,:])
    end

return vec_bell

end


function check_loop(type, current, vert)

    #find the next state
    dir = random_vector(type)    

    vmax =  2 * sqrt(3 / 4)
    vmin = -2 * sqrt(3 / 4)

    v = vmin + (vmax-vmin)*rand()
    a = current + v * dir

    while !checknewt(type,a)
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = current + v * dir    
    end
    current = a

    #check whether the state fulfills PPT(->cn), violates CHSH given Horodecki (->cs), violates CHSH with one of the 12 given measurement settings (->cm) 
    cn, cs, cm = 1, 0, 0

     if !checknewt(type, flip(type, a))
        cn = 0
        if type == 3 
            cm = (maximum(vert * a) > 2.0 + 1e-16 ) 
            ev = 4* [current[1]^2, current[2]^2, current[3]^2]
            cs = (ev[1]+ev[2] > 1+1e-16 || ev[3]+ev[2] > 1+1e-16 ||ev[1]+ev[3] > 1+1e-16)

        elseif type == 7
            cm = (maximum(vert * a[[1,2,4,5,7]]) > 2.0 + 1e-16 ) 
            ev =[2*a[2]^2 + 2*a[4]^2 + 2*a[5]^2 + 2*a[7]^2 - 2*sqrt(a[2]^4 + 2*a[2]^2*a[4]^2 + a[4]^4 + a[5]^4 + 8*a[2]*a[4]*a[5]*a[7] + a[7]^4 - 2*(a[2]^2 - a[4]^2)*a[5]^2 + 2*(a[2]^2 - a[4]^2 + a[5]^2)*a[7]^2), 2*a[2]^2 + 2*a[4]^2 + 2*a[5]^2 + 2*a[7]^2 + 2*sqrt(a[2]^4 + 2*a[2]^2*a[4]^2 + a[4]^4 + a[5]^4 + 8*a[2]*a[4]*a[5]*a[7] + a[7]^4 - 2*(a[2]^2 - a[4]^2)*a[5]^2 + 2*(a[2]^2 - a[4]^2 + a[5]^2)*a[7]^2), 4*a[1]^2]
            cs = (ev[1]+ev[2] > 1+1e-16 || ev[3]+ev[2] > 1+1e-16 || ev[1]+ev[3] > 1+1e-16) 
  
        elseif type == 9
            cm = (maximum(vert * a[[5,6,7,8,9]]) > 2.0 + 1e-16 ) 
            T = 2*[a[5] 0 a[6]; 0 a[7] 0; a[8] 0 a[9]]
            U = transpose(T)*T
            ev = eigvals(U)
            cs =  (ev[1]+ev[2] > 1+1e-16 || ev[3]+ev[2] > 1+1e-16 ||ev[1]+ev[3] > 1+1e-16) 
               
        elseif type == 15
            cm = (maximum(vert * a[[7,8,9,10,11,12,13,14,15]]) > 2.0 + 1e-16 ) 
            T = 2*[a[7] a[8] a[9]; a[10] a[11] a[12]; a[13] a[14] a[15]]
            U = transpose(T)*T
            ev = eigvals(U)
            cs = (ev[1]+ev[2] > 1+1e-16 || ev[3]+ev[2] > 1+1e-16 ||ev[1]+ev[3] > 1+1e-16) 
        end

    end

    return current, cn, cs, cm 
end

#additional function for parallelization
function res_loop(d,current,n,vert)
    r_ppt =0
    r_steinmetz =0
    r_meas12 =0
    for _ in 1:n
        current, cn, cs, cm = check_loop(d,current,vert)
        r_ppt += cn
        r_steinmetz += cs
        r_meas12 += cm
    end
    return current, r_ppt, r_steinmetz, r_meas12 
end


function test_par(type=3, rep=200, num=50000)

vert = bell_meas(type)
results = [0,0,0,0]
current = zeros(type)
    ppt = 0
    steinmetz = 0
    meas12 = 0
    @sync for _ in 1:rep
        Threads.@spawn begin
        current, r_ppt, r_steinmetz, r_meas12 = res_loop(type,current,num,vert)
        ppt += r_ppt
        steinmetz += r_steinmetz
        meas12 += r_meas12
    end 
end
results[1] += rep*num    
results[2] += ppt
results[3] += steinmetz
results[4] += meas12
str = string(results[1], ";; ", results[2]/results[1], ";  ", results[3]/results[1], ";  ", results[4]/results[1], ";  ", ppt, ";  ", steinmetz, ";  ", meas12)
println(str)

end


function inf_run_par(type=3, rep=20, num=5000)
    vert = bell_meas(type)
    results = [0,0,0,0]
    current = zeros(type)
    while true
        ppt = 0
        steinmetz = 0
        meas12 = 0
        @sync for _ in 1:rep
        Threads.@spawn begin
        current, r_ppt, r_steinmetz, r_meas12 = res_loop(type,current,num,vert)
        ppt += r_ppt
        steinmetz += r_steinmetz
        meas12 += r_meas12
        end 
    end
    results[1] += rep*num    
    results[2] += ppt
    results[3] += steinmetz
    results[4] += meas12
    str = string(results[1], ";; ", results[2]/results[1], ";  ", results[3]/results[1], ";  ", results[4]/results[1], ";  ", ppt, ";  ", steinmetz, ";  ", meas12)
    println(str)

    #io = open(string("results_steinmetz_12_d",type,".txt"), "a");
    #write(io, str, "\n");
    #close(io)
    
    end
end

#check runtime for one iteration
@time test_par(3,50,20000)


#First number:
#Bell diagonal  == 3
#X-state        == 7
#rebit          == 9
#full state     == 15

#inf_run_par(15,50,20000)
