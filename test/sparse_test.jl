import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MCTrajOpt
using BenchmarkTools
using Test
using SparseArrays

const MC = MCTrajOpt

struct MyVec{T,V} <: AbstractVector{T}
    data::V
    function MyVec(v::V) where V <: AbstractVector
        new{eltype(v), V}(v)
    end
end
Base.size(a::MyVec) = size(a.data)
Base.IndexStyle(::MyVec) = IndexLinear()
Base.getindex(a::MyVec, i::Integer) = v[i]

m = 100
n = 50

blocks = MC.BlockViews(m,n)
MC.setblock!(blocks, 1:3, 1:3)
@test blocks[1:3, 1:3] == 1:9
@test blocks.vec2blk[1:9] == MC.BlockID(1,1,3,3)

MC.setblock!(blocks, 4:6, 1:4)
@test blocks[4:6, 1:4] == 9 .+ (1:12)
@test blocks.vec2blk[9 .+ (1:12)] == MC.BlockID(4,1,3,4)

MC.setblock!(blocks,7:12,7:12)
@test blocks.len == 9 + 12 + 36

v = MC.NonzerosVector(zeros(9+12+36), blocks)
MC.getblock(blocks, v, 1:3, 1:3) .= 1
@test v[1:9] ≈ ones(9)

# Fill in the vector
A = spzeros(m,n)
B1 = randn(3,3)
v[1:3,1:3] = B1
A[1:3,1:3] = B1
@test v.data[1:9] ≈ vec(B1)
v[1:3,1:3] = 2
A[1:3,1:3] .= 2
@test v.data[1:9] ≈ fill(2,9) 
B2 = randn(4,3)
v[4:6,1:4] = B2'
A[4:6,1:4] = B2'
@test v[10:21] ≈ vec(B2')
B3 = randn(6,6)
v[7:12,7:12] = B3
A[7:12,7:12] = B3

rc = MC.getrc(blocks)
@test length(rc) == length(v)
@test rc[1] == (1,1)
@test rc[3] == (3,1)
@test rc[4] == (1,2)
@test rc[9] == (3,3)
@test rc[10] == (4,1)

# Cast to sparse matrix
A2 = sparse(v)
@test A2 ≈ A
@test nnz(A2) ≈ nnz(A)