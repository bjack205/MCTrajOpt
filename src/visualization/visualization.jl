using MeshCat, GeometryBasics, CoordinateTransformations, Rotations, Colors

function visualize!(vis, model, X::Vector{<:AbstractVector}, params::SimParams)
    anim = MeshCat.Animation(floor(Int,1/params.h))
    for k = 1:params.N
        atframe(anim, k) do
            visualize!(vis, model, X[k])
        end
    end
    setanimation!(vis, anim)
end

function launchvis(model::RobotArm, x0)
    vis = Visualizer()
    for (j, body) in enumerate(model.geom)
        geom = geometry(body)
        mat = MeshPhongMaterial(color=body.color)      
        name = "body" * string(j)
        setobject!(vis[name], geom, mat)
        setobject!(vis[name]["com"], Triad())
        settransform!(vis[name]["com"], LinearMap(I(3)*0.25))
    end
    open(vis)
    visualize!(vis, model, x0)
    return vis
end

function visualize!(vis, model::RobotArm, x)
    for j = 1:model.numlinks
        r = gettran(model, x, j)
        q = getquat(model, x, j)
        T = compose(Translation(r), LinearMap(UnitQuaternion(q)))
        name = "body" * string(j)
        settransform!(vis[name], T)
    end
end

function launchvis(body::DoublePendulum, x0; geom=nothing)
    vis = Visualizer()
    if isnothing(geom)
        geom1 = Cylinder(Point(body.joint0.p2), Point(body.joint1.p1), 0.2)
        geom2 = Cylinder(Point(body.joint1.p2), -Point(body.joint1.p2), 0.2)
    else
        geom1 = geometry(geom[1])
        geom2 = geometry(geom[2])
    end
    setobject!(vis["body1"], geom1, MeshPhongMaterial(color=colorant"green"))
    setobject!(vis["body2"], geom2, MeshPhongMaterial(color=colorant"green"))
    setobject!(vis["body1"]["com"], Triad())
    setobject!(vis["body2"]["com"], Triad())
    settransform!(vis["body1"]["com"], LinearMap(I*1/4))
    settransform!(vis["body2"]["com"], LinearMap(I*1/4))
    let x = x0
        x1,x2 = MCTrajOpt.splitstate(body, x)
        ri,qi = SA[1,2,3], SA[4,5,6,7]
        r1,r2 = x1[ri], x2[ri]
        q1,q2 = x1[qi], x2[qi]
        settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
        settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
    end
    open(vis)
    return vis
end

function visualize!(vis, model::TwoBody, x::AbstractVector{<:Real})
    x1,x2 = MCTrajOpt.splitstate(model, x)
    ri,qi = SA[1,2,3], SA[4,5,6,7]
    r1,r2 = x1[ri], x2[ri]
    q1,q2 = x1[qi], x2[qi]
    settransform!(vis["body1"], compose(Translation(r1), LinearMap(UnitQuaternion(q1))))
    settransform!(vis["body2"], compose(Translation(r2), LinearMap(UnitQuaternion(q2))))
end
