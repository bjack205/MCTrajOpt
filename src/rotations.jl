function hat(ω)
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end

function L(q)
    [q[1] -q[2:4]'; q[2:4] q[1]*I + hat(q[SA[2,3,4]])]
end

function R(q)
    [q[1] -q[2:4]'; q[2:4] q[1]*I - hat(q[SA[2,3,4]])]
end

const Hmat = SA[
    0 0 0
    1 0 0
    0 1 0
    0 0 1
]
const Tmat = Diagonal(SA[1,-1,-1,-1])

function G(q)
    return L(q)*Hmat
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