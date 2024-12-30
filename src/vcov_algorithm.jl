struct CovarianceTensor{T}
    dims::Vector{Int}
    vec::Vector{Union{T,CovarianceTensor{T}}}
end

Base.size(ct::CovarianceTensor) = Tuple(ct.dims)
function Base.getindex(ct::CovarianceTensor{T}, I::Vararg{Int,N}) where {T,N}
    if ct.vec[I[1]] isa CovarianceTensor
        return getindex(ct.vec[I[1]], I[2:end]...)
    else
        return ct.vec[I[1]]
    end
end

function Base.setindex!(ct::CovarianceTensor{T}, value, I::Vararg{Int,N}) where {T,N}
    if length(ct.dims) != N
        throw(ArgumentError("Indexing dimensions do not match CovarianceTensor dimensions"))
    end
    if N == 1
        ct.vec[I[1]] = T(value)
        return value
    end

    if ct.vec[I[1]] isa T
        ct.vec[I[1]] = ctzeros(T, ct.dims[2:end]...)
    end
    return setindex!(ct.vec[I[1]], value, I[2:end]...)
end

ctzeros(T, I::Vararg{Int,N}) where {N} =
    CovarianceTensor{T}([I...], Vector{Union{T,CovarianceTensor{T}}}(undef, I[1]) .= zero(T))

function scale!(ct::CovarianceTensor{T}, scalar) where {T}
    for i in 1:length(ct.vec)
        if ct.vec[i] isa CovarianceTensor
            scale!(ct.vec[i], scalar)
        else
            ct.vec[i] *= scalar
        end
    end
    return ct
end

function Base.Array(ct::CovarianceTensor{T}) where {T}
    dims = Tuple(ct.dims)
    arr = zeros(T, dims)
    for I in Tuple.(CartesianIndices(dims))
        arr[I...] = ct[I...]
    end
    return arr
end

function estimate_giv(
    qmat,
    Cpts,
    Cts,
    Smat,
    exclmat,
    ::Val{:vcov};
    guess = nothing,
    quiet = false,
    solver_options = (;),
)
    N, T, Nmom = size(Cts)
    Cp = reshape(Cpts, N * T, Nmom)
    if isnothing(guess)
        if !quiet
            @info "Initial guess is not provided. Using OLS estimate as initial guess."
        end
        guess = (Cp' * Cp) \ (Cp' * vec(qmat))
    end
    Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp = compuate_covariance_tensors(qmat, Cpts, Cts)
    if norm(var(Smat; dims = 2)) .> eps(eltype(Smat)) && !quiet
        @warn "Weights are time-varying. Averages across time are used for the weighting scheme."
    end
    S = vec(mean(Smat; dims = 2))
    err0 = mean_vcov_err(guess, Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp, S, exclmat)
    if length(err0) != Nmom
        throw(ArgumentError("The number of moment conditions is not equal to the number of initial guess."))
    end
    res = nlsolve(
        x -> mean_vcov_err(x, Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp, S, exclmat),
        guess;
        solver_options...,
    )
    converged = res.f_converged
    if !converged && !quiet
        @warn "The estimation did not converge."
    end
    return res.zero, res.f_converged
end

function compuate_covariance_tensors(qmat::AbstractArray{F,2}, Cpts, Cts) where {F}
    N, T, Nmom = size(Cpts)
    Cqq = ctzeros(F, N, Nmom, N)
    CqCp = ctzeros(F, N, Nmom, Nmom, N)
    CCpq = ctzeros(F, N, Nmom, Nmom, N)
    CCpCp = ctzeros(F, N, Nmom, Nmom, N, Nmom)

    qq = ctzeros(F, N)
    Cpq = ctzeros(F, N, Nmom)
    CpCp = ctzeros(F, N, Nmom, Nmom)
    @threads for i in 1:N
        for t in 1:T
            qq[i] += qmat[i, t] * qmat[i, t]
            for g in 1:Nmom
                !iszero(Cts[i, t, g]) || continue
                Cq_ig = qmat[i, t] * Cts[i, t, g]
                for j in 1:N
                    Cqq[i, g, j] += Cq_ig * qmat[j, t]
                end
                for (k, j) in Tuple.(CartesianIndices((1:Nmom, 1:N)))
                    !iszero(Cpts[j, t, k]) || continue
                    CqCp[i, g, k, j] += Cq_ig * Cpts[j, t, k]
                end
                for k in 1:Nmom
                    !iszero(Cpts[i, t, k]) || continue
                    for j in 1:N
                        CCpq[i, g, k, j] += Cts[i, t, g] * Cpts[i, t, k] * qmat[j, t]
                    end
                    for (k2, j) in Tuple.(CartesianIndices((1:Nmom, 1:N)))
                        !iszero(Cpts[j, t, k2]) || continue
                        CCpCp[i, g, k, j, k2] += Cts[i, t, g] * Cpts[i, t, k] * Cpts[j, t, k2]
                    end

                    CpCp[i, g, k] += Cpts[i, t, g] * Cpts[i, t, k]
                end
                Cpq[i, g] += Cpts[i, t, g] * qmat[i, t]
            end
        end
    end
    for ct in [Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp]
        scale!(ct, 1 / (T - 1))
    end
    return Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp
end

function compute_u_variance(ζ, qq, Cpq, CpCp::CovarianceTensor{T}) where {T}
    N, Nmom = size(Cpq)
    σu²vec = zeros(T, N)
    @threads for i in 1:N
        σu²vec[i] = qq[i]
        for k in 1:Nmom
            !iszero(Cpq[i, k]) || continue
            σu²vec[i] += 2 * Cpq[i, k] * ζ[k]
            for j in 1:Nmom
                !iszero(CpCp[i, k, j]) || continue
                σu²vec[i] += CpCp[i, k, j] * ζ[k] * ζ[j]
            end
        end
    end
    return σu²vec
end

function compute_err_ig(ζ, Cqq, CqCp, CCpq, CCpCp, exclmat)
    N, Nmom, N = size(Cqq)
    err = ctzeros(eltype(ζ), N, Nmom, N)
    @threads for i in 1:N
        for g in 1:Nmom
            !iszero(Cqq[i, g, 1]) || continue # Cts[i, g] is zero for all t
            for j in 1:N
                !exclmat[i, j] || continue
                err[i, g, j] += Cqq[i, g, j]
                for k in 1:Nmom
                    CqCp_igkj = CqCp[i, g, k, j]
                    !iszero(CqCp_igkj) || continue
                    err[i, g, j] += CqCp_igkj * ζ[k]
                end
                for k in 1:Nmom
                    CCpq_igkj = CCpq[i, g, k, j]
                    !iszero(CCpq_igkj) || continue
                    err[i, g, j] += CCpq_igkj * ζ[k]
                end
                for k in 1:Nmom
                    for k2 in 1:Nmom
                        CCpCp_igkjk2 = CCpCp[i, g, k, j, k2]
                        !iszero(CCpCp_igkjk2) || continue
                        err[i, g, j] += CCpCp_igkjk2 * ζ[k] * ζ[k2]
                    end
                end
            end
        end
    end
    return err
end

function mean_vcov_err(ζ, Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp, S, exclmat)
    N, Nmom = size(Cpq)
    σu²vec = compute_u_variance(ζ, qq, Cpq, CpCp)
    err_ig = compute_err_ig(ζ, Cqq, CqCp, CCpq, CCpCp, exclmat)
    err = zeros(Nmom)
    precision = 1 ./ σu²vec
    weightsum = zeros(Nmom)
    @threads for g in 1:Nmom
        for i in 1:N
            for j in (i+1):N
                !exclmat[i, j] || continue
                err_ig_igj = err_ig[i, g, j]
                !iszero(err_ig_igj) || continue
                weight = S[j] * precision[i]
                err[g] += err_ig_igj * weight
                weightsum[g] += weight
            end
        end
        for j in 1:N
            for i in 1:(j-1)
                !exclmat[i, j] || continue
                err_ig_jgi = err_ig[j, g, i]
                !iszero(err_ig_jgi) || continue
                weight = S[i] * precision[j]
                err[g] += err_ig_jgi * weight
                weightsum[g] += weight
            end
        end
    end
    weightsum ./= sum(abs.(weightsum))
    err ./= weightsum
    return err
end
