module Types

export memref, i1, i8, i16, i32, i64, f32, f64, index, tensor

using MLIR
using MLIR.IR: Value, Attribute, MLIRType, get_value, get_result, Operation, Convertible
using MLIR.API: mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize
import MLIR.IR: MLIRValueTrait

abstract type MLIRArrayLike{T, N} <: AbstractArray{T, N} end
MLIRValueTrait(::Type{<:MLIRArrayLike}) = Convertible()
Base.show(io::IO, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")

struct MLIRMemref{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
MLIR.IR.MLIRType(::Type{MLIRMemref{T, N}}) where {T, N} = MLIRType(Array{T, N})
const memref = MLIRMemref

struct MLIRTensor{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
MLIR.IR.MLIRType(::Type{MLIRTensor{T, N}}) where {T, N} = mlirRankedTensorTypeGet(
    N,
    Int[mlirShapedTypeGetDynamicSize() for _ in 1:N],
    MLIRType(T),
    Attribute()) |> MLIRType
const tensor = MLIRTensor

struct MLIRInteger{N} <: Integer
    value::Value
    MLIRInteger{N}(i::Value) where {N} = new(i)
    MLIRInteger{N}(i::Operation) where {N} = new(get_result(i))
end
MLIR.IR.MLIRType(::Type{MLIRInteger{N}}) where {N} = MLIRType(mlirIntegerTypeGet(MLIR.IR.context(), N))
MLIRValueTrait(::Type{<:MLIRInteger}) = Convertible()

const i1 = MLIRInteger{1}
@noinline i1(::Bool)::i1 = Jojo.new_intrinsic()

const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

abstract type MLIRFloat <: AbstractFloat end
MLIRValueTrait(::Type{<:MLIRFloat}) = Convertible()

struct MLIRF64 <: MLIRFloat
    value::Value
end
const f64 = MLIRF64

struct MLIRF32 <: MLIRFloat
    value::Value
end
const f32 = MLIRF32


struct MLIRIndex <: Integer
    value::Value
end
const index = MLIRIndex
MLIR.IR.MLIRType(::Type{MLIRIndex}) = MLIR.IR.IndexType()
MLIRValueTrait(::Type{<:MLIRIndex}) = Convertible()

end # Types
