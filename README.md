# Jojo.jl
> Generate MLIR from Julia code.

## Overview

`script.jl` contains an example of generating MLIR for some simple Julia functions. Here's a quick overview of the main features:

**Create regular Julia types and define a mapping to MLIR by overwriting MLIR.jl's `IR.MLIRType` and `IR.MLIRValueTrait`.**

```julia
using MLIR: IR, API

struct MLIRInteger{N} <: Integer
    value::IR.Value
    MLIRInteger{N}(i::IR.Value) where {N} = new(i)
end
IR.MLIRType(::Type{MLIRInteger{N}}) where {N} = IR.MLIRType(
    API.mlirIntegerTypeGet(IR.context(), N))
IR.MLIRValueTrait(::Type{<:MLIRInteger}) = IR.Convertible()

i64 = MLIRInteger{64}
```
As an example, `src/ValueTypes.jl` contains definitions for a few MLIR types.

**Specialize functions on your type to generate MLIR operations.**
    
```julia
using MLIR.Dialects: arith
Jojo.@mlirfunction function Base.:+(a::T, b::T)::T where {T <: MLIRInteger}
    T(IR.get_result(arith.addi(a, b)))
end
```

**Generate MLIR for regular Julia code.**

```julia
Jojo.code_mlir(Tuple{i64, i64}) do a, b
    a+b
end
```
```mlir
module {
    func.func @"#17"(%arg0: i64, %arg1: i64) -> i64 attributes {llvm.emit_c_interface} {
        %0 = arith.addi %arg0, %arg1 : i64
        return %0 : i64
    }
}
```


## Installation
This code currently depends on forks of packages that are not yet registered or upstreamed ([CodeInfoTools.jl#21](https://github.com/JuliaCompilerPlugins/CodeInfoTools.jl/pull/21) and [MLIR.jl](https://github.com/JuliaLabs/MLIR.jl/tree/jm/jojo)). A `Manifest.toml` is checked in for easy reproducibility. The Julia version used was:
```
Julia Version 1.11.0-DEV.1461
Commit fc062919c3f (2024-02-03 02:49 UTC)
```
## Internals

`Jojo.jl` builds on top of `MLIR.jl` and especially [this demo](https://github.com/JuliaLabs/MLIR.jl/blob/jm/jojo/examples/brutus.jl), written by Pangoraw, for generating MLIR from Julia.

### `@mlirfunction`
`Jojo.@mlirfunction` makes sure that function definitions are not const-propped or inlined by the Julia compiler. This makes them always clearly visible Julia IR.
The code in `src/Jojo.jl` loops over all statements in the IR, if it encounters a call to a function that was defined with `@mlirfunction`, it executes the function.
A small but important detail is that calls to `MLIR.IR.create_operation` are overlayed using `CassetteOverlay` to automatically insert the created operations into the MLIR module that's currently being generated.

### Booleans
Since Julia doesn't have a concept of bool-like user-defined types. It's not possible to define a type yourself that acts like a boolean. For example the `i1` type from the builtin MLIR dialect.
To circumvent this, Jojo uses a custom Abstract Interpreter that inserts conversions from user defined types to `Bool` when it encounters `gotoifnot` statements in the IR (see `insert_bool_conversions_pass` in `src/abstract.jl`).

To enable this conversion for a user defined type, `Jojo.BoolTrait` should be specialized for that type to return `Jojo.Boollike()`.

### `emit_region=true`
By providing a `emit_region=true` kwarg to `Jojo.code_mlir`, Jojo won't wrap the generated MLIR in a module but rather return the region as is.
This is useful when you want to generate MLIR from Julia code within a `@mlirfunction`.
An example would be generating the body of a `linalg.generic` operation. These examples are from an in-development version using a more recent MLIR version and require some more plumbing to make things work, but should still give an idea of what's possible.
```julia
@mlirfunction function mul!(Y::tensor{T, 2}, A::tensor{T, 2}, B::tensor{T, 2})::tensor{T, 2} where T
    indexing_maps = ...
    iterator_types = ...

    matmul_region = @nonoverlay Brutus.code_mlir((a, b, y)->linalgyield(y+(a*b)), Tuple{T, T, T}; emit_region=true, ignore_returns=true)

    op = linalg.generic(
        [A, B],
        [Y],
        result_tensors=MLIRType[MLIRType(typeof(Y))];
        indexing_maps,
        iterator_types,
        region=matmul_region
    )
    return tensor{T, 2}(IR.get_result(op))
end
```
```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @f(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>, %arg2: tensor<?x?xi64>) -> tensor<?x?xi64> attributes {llvm.emit_c_interface} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<?x?xi64>, tensor<?x?xi64>) outs(%arg0 : tensor<?x?xi64>) {
  ^bb0(%in: i64, %in_0: i64, %out: i64):
    %1 = arith.muli %in, %in_0 : i64
    %2 = arith.addi %out, %1 : i64
    linalg.yield %2 : i64
  } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}
```

Or even, using higher order functions:
```julia
Jojo.code_mlir(Tuple{}) do
    named_sequence() do op
        matched = structured_match(op, "linalg.generic")
        tiled = structured_tile_using_for(matched, (0, 0, 1))
        tiled = structured_tile_using_for(tiled, (0, 1, 0))
        yield()
    end
end
```
```mlir
func.func @"#19"() attributes {llvm.emit_c_interface} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0[0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op[0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
  return
}
```
In this example, all the functions used in the `named_sequence` as well as `named_sequence` itself are defined with `@mlirfunction`.
