const I3 = SA[1 0 0; 0 1 0; 0 0 1]
const I4 = SA[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

function hat(ω)
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end

function L(q)
    SA[
        q[1] -q[2] -q[3] -q[4]
        q[2] q[1] -q[4] q[3]
        q[3] q[4] q[1] -q[2]
        q[4] -q[3] q[2] q[1]
    ]
end

function R(q)
    SA[
        q[1] -q[2] -q[3] -q[4]
        q[2] q[1] q[4] -q[3]
        q[3] -q[4] q[1] q[2]
        q[4] q[3] -q[2] q[1]
    ]
end

const Hmat = SA[
    0 0 0
    1 0 0
    0 1 0
    0 0 1
]
const Tmat = Diagonal(SA[1,-1,-1,-1])

function G(q)
    return L(q)*Hmat/2
end

"""
Jacobian of `G(q)'b` wrt quaternion `q`, and `b` has length 4.
"""
function ∇G(q, b)
    # I3 = SA[1 0 0; 0 1 0; 0 0 1]
    # return -I3 * (q'b)
    Hmat'R(b)*Tmat/2
end

function ∇G2(q, b)
    I3 = SA[1 0 0; 0 1 0; 0 0 1]
    return -I3 * (q'b/4)
    # Hmat'R(b)*Tmat
end

function compose_states(x1,x2)
    qi = SA[4,5,6,7]
    r3 = SA[x1[1] + x2[1], x1[2] + x2[2], x1[3] + x2[3]]
    q3 = L(x1[qi]) * x2[qi]
    x3 = SA[
        r3[1], r3[2], r3[3],
        q3[1], q3[2], q3[3], q3[4],
    ]
    return x3
end

function errstate_jacobian(x)
    qi = SA[4,5,6,7]
    q = x[qi]
    G0 = G(q)
    SA[
        1 0 0 0 0 0
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 0 G0[1] G0[5] G0[9]
        0 0 0 G0[2] G0[6] G0[10]
        0 0 0 G0[3] G0[7] G0[11]
        0 0 0 G0[4] G0[8] G0[12]
    ]
end

"""
Jacobian of `errstate_jacobian(x)'b` for `b` of length 7. Size is 6×7
"""
function ∇errstate_jacobian(x, b)
    qi = SA[4,5,6,7]
    q = x[qi]
    bq = b[qi]
    G = ∇G(q, bq)
    SA[
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 G[1] G[4] G[7] G[10]
        0 0 0 G[2] G[5] G[8] G[11]
        0 0 0 G[3] G[6] G[9] G[12]
    ]
end

function ∇errstate_jacobian2(x, b)
    qi = SA[4,5,6,7]
    q = x[qi]
    bq = b[qi]
    G = ∇G2(q, bq)
    SA[
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 G[1] G[4] G[7]
        0 0 0 G[2] G[5] G[8]
        0 0 0 G[3] G[6] G[9]
    ]
end

function cayleymap(g)
    g /= 2
    M = 1/sqrt(1+g'g)
    SA[M, M*g[1], M*g[2], M*g[3]]
end

function err2fullstate(x)
    ϕ = x[SA[4,5,6]]
    q = cayleymap(ϕ) 
    SA[x[1], x[2], x[3], q[1], q[2], q[3], q[4]]
end

function kinematics(x,ν)
    q = x[SA[4,5,6,7]]
    v = ν[SA[1,2,3]]
    ω = ν[SA[4,5,6]]
    qdot = 0.5 * L(q) * Hmat * ω
    SA[
        v[1],v[2],v[3],
        qdot[1],qdot[2],qdot[3],qdot[4]
    ]
end

function D1K(x,ν)
    ω = ν[SA[4,5,6]]
    D = R(Hmat*ω) / 2
    SA[
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 D[1] D[5] D[9]  D[13]
        0 0 0 D[2] D[6] D[10] D[14]
        0 0 0 D[3] D[7] D[11] D[15]
        0 0 0 D[4] D[8] D[12] D[16]
    ]
end

function D2K(x,ν)
    q = x[SA[4,5,6,7]]
    D = L(q) * Hmat / 2
    SA[
        1 0 0 0 0 0
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 0 D[1] D[5] D[9] 
        0 0 0 D[2] D[6] D[10]
        0 0 0 D[3] D[7] D[11]
        0 0 0 D[4] D[8] D[12]
    ]
end

function inv_kinematics(x,xdot)
    q = x[SA[4,5,6,7]]
    rdot = xdot[SA[1,2,3]]
    qdot = xdot[SA[4,5,6,7]]
    v = rdot
    ω = 2Hmat'*L(q)'qdot
    return [v; ω]
end

function D1Kinv(x,xdot)
    qdot = xdot[SA[4,5,6,7]]
    J = 2Hmat'R(qdot)*Tmat
    SA[
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 J[1] J[4] J[7] J[10]
        0 0 0 J[2] J[5] J[8] J[11]
        0 0 0 J[3] J[6] J[9] J[12]
    ]
end

function D2Kinv(x,xdot)
    q = x[SA[4,5,6,7]]
    J = 2Hmat'L(q)'
    SA[
        1 0 0 0 0 0 0
        0 1 0 0 0 0 0
        0 0 1 0 0 0 0
        0 0 0 J[1] J[4] J[7] J[10]
        0 0 0 J[2] J[5] J[8] J[11]
        0 0 0 J[3] J[6] J[9] J[12]
    ]
end

function Amat(q)
    w,x,y,z = q[1],q[2],q[3],q[4]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    xy = (x * y)
    zw = (w * z)
    xz = (x * z)
    yw = (y * w)
    yz = (y * z)
    xw = (w * x)
    SA[
        ww + xx - yy - zz  2(xy - zw)         2(xz + yw)
        2(xy + zw)         ww - xx + yy - zz  2(yz - xw)
        2(xz - yw)         2(yz + xw)         ww - xx - yy + zz
    ]
end

function ∇rot(q, r)
    rhat = SA[0, r[1], r[2], r[3]]
    2*Hmat'R(q)'R(rhat)
end

"""
Jacobian of `∇rot(q, r,)'b` wrt `q` (note the transpose)
"""
function ∇²rot(q, r, b)
    rhat = SA[0, r[1], r[2], r[3]]
    2*R(rhat)'L(Hmat*b)
end

function expm(ϕ)
    θ = norm(ϕ)
    sθ,cθ = sincos(θ/2)
    M = sinc(θ/π/2)/2
    SA[cθ, ϕ[1]*M, ϕ[2]*M, ϕ[3]*M]
end

function logm(q)
    w = q[1]
    x = q[2]
    y = q[3]
    z = q[4]

    θ = sqrt(x^2 + y^2 + z^2)
    if θ > 1e-6
        M = atan(θ, w)/θ
    else
        M = (1-θ^2/(3w^2))/w
    end
    2*M*SA[x,y,z]
end