using GeometryBasics
using Colors
using FixedPointNumbers

const DENSITIES = Dict(
    :aluminum => 2.7 * 1000,
    :steel => 8.05 * 1000
)

struct CylindricalBody
    density::Float64
    radius::Float64
    length::Float64
    color::RGB{FixedPointNumbers.N0f8}
end

function CylindricalBody(radius::Real, length::Real, mat::Symbol; kwargs...)
    CylindricalBody(radius, length, density=DENSITIES[mat]; kwargs...)
end

function CylindricalBody(radius::Real, length::Real; 
    density::Real=DENSITIES[:aluminum], color=colorant"blue"
)
    CylindricalBody(Float64(density), Float64(radius), Float64(length), color)
end

mass(body::CylindricalBody) = π*body.radius^2 * body.length * body.density

function inertiatensor(body::CylindricalBody)
    r,h = body.radius, body.length
    m = mass(body)
    Ixx = Iyy = m*(3*r^2 + h^2) / 12
    Izz = m*r^2/2
    SA[
        Ixx 0 0
        0 Iyy 0
        0 0 Izz
    ]
end

function geometry(body::CylindricalBody)
    h = body.length
    Cylinder(Point3(0,0,-h/2), Point3(0,0,h/2), body.radius)
end

function RigidBody(body::CylindricalBody)
    RigidBody(mass(body), inertiatensor(body))
end