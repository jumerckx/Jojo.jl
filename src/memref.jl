struct MemRef{T,N} <: DenseArray{T, N}
    allocated_pointer::Ptr{T}
    aligned_pointer::Ptr{T}
    offset::Int
    sizes::NTuple{N, Int}
    strides::NTuple{N, Int}
    data::Array{T, N}
end
function MemRef(a::Array{T,N}) where {T,N}
    @assert isbitstype(T) "Array element type should be isbits, got $T."
    allocated_pointer = a.ref.mem.ptr
    aligned_pointer = a.ref.ptr_or_offset
    offset = Int((aligned_pointer - allocated_pointer)//sizeof(T))
    @assert offset == 0 "Arrays with Memoryref offset are, as of yet, unsupported."
    sizes = size(a)
    strides = Tuple([1, cumprod(size(a))[1:end-1]...])
    
    return MemRef{T,N}(
        allocated_pointer,
        aligned_pointer,
        offset,
        sizes,
        strides,
        a,
        )
end

Base.show(io::IO, A::Jojo.MemRef{T, N}) where {T, N} = print(io, "Jojo.MemRef{$T,$N} (size $(join(A.sizes, "×")))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, X::Jojo.MemRef) = show(io, X)
Base.size(A::MemRef) = A.sizes
