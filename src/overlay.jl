using CassetteOverlay, MacroTools
import MLIR.IR.create_operation

@noinline new_intrinsic()::T where T = Base.inferencebarrier(true) && error("Intrinsics should be compiled to MLIR!")

@MethodTable MLIRCompilation
mlircompilationpass = @overlaypass MLIRCompilation
@inline (::typeof(mlircompilationpass))(::typeof(Base.typename), @nospecialize args...) = 
    Base.typename(args...)

@overlay MLIRCompilation function IR.create_operation(
        name, loc;
        results=nothing,
        operands=nothing,
        owned_regions=nothing,
        successors=nothing,
        attributes=nothing,
        result_inference=isnothing(results))
    @info "overlaid create_operation for $name"
    op = @nonoverlay IR.create_operation(
        name, loc;
        results,
        operands,
        owned_regions,
        successors,
        attributes,
        result_inference)
    push!(currentblock(codegencontext()), op)

    cg = codegencontext()

    return op
end

function mlirfunction_(expr)
    dict = splitdef(expr)
    if !(:rtype in keys(dict))
        error("Result type should be specified in function definition, `f(args)::T = ...` or `function f(args)::T ...`")
    end
    rtype = dict[:rtype]

    # since the methodtable shouldn't be escaped, we create an alias that escapes and binds to the real deal.
    methodtable = gensym(:MLIRCompilation)

    expr = Base.Experimental.overlay_def!(methodtable, combinedef(dict))

    return quote
        $(esc(methodtable)) = MLIRCompilation

        @noinline function $(esc(dict[:name]))($(esc.(dict[:args])...); $(esc.(dict[:kwargs])...))::$(esc(rtype)) where {$(esc.(dict[:whereparams])...)}
            new_intrinsic()
        end

        $(esc(expr))
    end
end

"""
    @mlirfunction [function def]

Will define the provided function definition twice. Once replacing the body with a call to
`new_intrinsic()`, and once with the body as-is, but registered in the `MLIRCompilation` method table.
"""
macro mlirfunction(f)
    f = macroexpand(__module__, f)
    Base.is_function_def(f) || error("@mlirfunction requires a function definition")
    mlirfunction_(f)
end