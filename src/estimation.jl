function estimate_giv(
    qmat,
    Cpts,
    Cts,
    Smat,
    exclmat,
    ::A;
    guess = nothing,
    quiet = false,
    solver_options = (;),
) where {A<:Union{Val{:iv},Val{:debiased_ols}}}
    N, T, Nmom = size(Cts)
    Cp = reshape(Cpts, N * T, Nmom)
    if isnothing(guess)
        if !quiet
            @info "Initial guess is not provided. Using OLS estimate as initial guess."
        end
        guess = (Cp' * Cp) \ (Cp' * vec(qmat))
    end
    err0 = mean_moment_conditions(guess, qmat, Cpts, Cts, Smat, exclmat, A())
    if length(err0) != Nmom
        throw(ArgumentError("The number of moment conditions is not equal to the number of initial guess."))
    end
    res = nlsolve(
        x -> mean_moment_conditions(x, qmat, Cpts, Cts, Smat, exclmat, A()),
        guess;
        solver_options...,
    )
    converged = res.f_converged
    ζ̂ = res.zero
    if !converged && !quiet
        @warn "The estimation did not converge."
    end
    return ζ̂, converged
end

function moment_conditions(ζ, qmat, Cpts, Cts, Smat, exclmat, ::Val{:iv})
    Nmom = length(ζ)
    N, T = size(qmat)
    u = vec(qmat) + reshape(Cpts, N * T, Nmom) * ζ
    umat = reshape(u, N, T)

    σu²vec = [var(umat[i, :]; mean = zero(eltype(umat))) for i in 1:N]
    precision = 1 ./ σu²vec
    precision ./= sum(precision)
    err = zeros(eltype(ζ), Nmom, T)
    weightsum = zeros(eltype(ζ), Nmom, T) # equal weight moment conditions for numerical stability
    @threads for (imom, t) in Tuple.(CartesianIndices((Nmom, T)))
        for i in 1:N
            if iszero(Cts[i, t, imom])
                continue
            end
            weight = Cts[i, t, imom] * precision[i]
            for j in (i+1):N
                !exclmat[i, j] || continue
                err[imom, t] += umat[i, t] * umat[j, t] * weight * Smat[j, t]
            end
            weightsum[imom, t] += precision[i]
        end

        for j in 1:N
            if iszero(Cts[j, t, imom])
                continue
            end
            weight = Cts[j, t, imom] * precision[j]
            for i in 1:(j-1)
                !exclmat[i, j] || continue
                err[imom, t] += umat[i, t] * umat[j, t] * weight * Smat[i, t]
            end
            weightsum[imom, t] += precision[j]
        end
    end
    # err ./= mean(weightsum, dims = 2) # equal weight moment conditions for numerical stability
    momweight = sum(abs.(weightsum); dims = 2)
    momweight = momweight ./ sum(momweight)
    err ./= momweight # equal weight moment conditions for numerical stability
    return err
end

function moment_conditions(ζ, qmat, Cpts, Cts, Smat, exclmat, ::Val{:debiased_ols})
    Nmom = length(ζ)
    N, T = size(qmat)
    u = vec(qmat) + reshape(Cpts, N * T, Nmom) * ζ
    umat = reshape(u, N, T)

    σu²vec = [var(umat[i, :]; mean = zero(eltype(umat))) for i in 1:N]
    precision = 1 ./ σu²vec
    precision = precision ./ sum(precision)
    err = zeros(eltype(ζ), Nmom, T)
    ζSvec = solve_aggregate_elasticity(ζ, Cts, Smat)
    weightsum = zeros(eltype(ζ), Nmom, T) # equal weight moment conditions for numerical stability
    @threads for (imom, t) in Tuple.(CartesianIndices((Nmom, T)))
        for i in 1:N
            if iszero(Cts[i, t, imom])
                continue
            end
            weight = precision[i]
            uCp = umat[i, t] * Cpts[i, t, imom]
            CSσ² = umat[i, t]^2 * Cts[i, t, imom] * Smat[i, t]
            # alternatively, use estimated variance
            # CSσ² = σu²vec[i,t] *(T-1)/T * Cts[i, t, imom] * Smat[i, t]
            err[imom, t] += weight * (uCp - CSσ² / ζSvec[t])
            weightsum[imom, t] += weight
        end
    end
    momweight = sum(abs.(weightsum); dims = 2)
    momweight = momweight ./ sum(momweight)
    err ./= momweight # equal weight moment conditions for numerical stability
    return err
end

mean_moment_conditions(ζ, args...) = vec(mean(moment_conditions(ζ, args...); dims = 2))

function estimate_loading_on_η(x, η, cholη = cholesky(η' * η))
    λ = cholη \ (η' * x)
    return λ
end

function residualize_on_η(x, η, cholη = cholesky(η' * η))
    λ = estimate_loading_on_η(x, η, cholη)
    return x - η * λ, λ
end

function solve_aggregate_elasticity(ζ, Cts, Smat)
    N, T, Nmom = size(Cts)
    Cζmat = sum(Cts .* reshape(ζ, 1, 1, Nmom); dims = 3)
    ζSvec = vec(sum(Smat .* Cζmat; dims = 1))
    return ζSvec
end

function solve_vcov(ζ, umat, Smat, Cts)
    Nmom = length(ζ)
    N, T, Nmom = size(Cts)
    σu²vec = [var(umat[i, :]; mean = zero(eltype(umat))) for i in 1:N]
    ζSvec = solve_aggregate_elasticity(ζ, Cts, Smat)
    Mvec = 1 ./ ζSvec
    Vdiag = zeros(N * (N - 1) ÷ 2)
    D = zeros(N * (N - 1) ÷ 2, Nmom, T)
    ind = 0
    for i in 1:N
        for j in i+1:N
            ind += 1
            Vdiag[ind] = σu²vec[i] * σu²vec[j]
            for (imom, t) in Tuple.(CartesianIndices((Nmom, T)))
                D[ind, imom, t] =
                    σu²vec[j] * Smat[j, t] * Cts[i, t, imom] +
                    σu²vec[i] * Smat[i, t] * Cts[j, t, imom]
                D[ind, imom, t] *= Mvec[t]
            end
        end
    end
    Vinv = inv(Diagonal(Vdiag))
    DVinvD = mean([D[:, :, t]' * Vinv * D[:, :, t] for t in 1:T])
    Σζ = inv(DVinvD) / T
    return σu²vec, Σζ
end


"""
Solve the OLS covariance matrix for the GIV estimation.

Code aggressively optimized by ChatGPT o1 at the sacrifice of readability.

More readable code:

```julia
function solve_ols_vcov(umat, ηts)
    N, T, Nmom = size(ηts)
    σu²vec = [var(umat[i, :]; mean = zero(eltype(umat))) for i in 1:N]
    ηηvec = [ηts[i, :, :]' * ηts[i, :, :] / T for i in 1:N]
    bread = inv(sum(ηηvec))
    meat = sum(ηηvec .* σu²vec)
    vcov_ols = bread * meat * bread / T
    return vcov_ols
end
```
"""
function solve_ols_vcov(umat, ηts)
    N, T, Nmom = size(ηts)

    # Initialize per-thread accumulators to avoid data races
    nthreads = Threads.nthreads()
    breads = [zeros(Nmom, Nmom) for _ in 1:nthreads]
    meats = [zeros(Nmom, Nmom) for _ in 1:nthreads]

    @threads for i in 1:N
        threadid = Threads.threadid()
        σu² = var(@view umat[i, :]; mean = zero(eltype(umat)))

        # Skip computations if variance is zero
        if σu² == 0
            continue
        end

        # Use views to avoid copying data
        @views ηi = ηts[i, :, :]

        # Identify non-zero columns to exploit sparsity
        zero_cols = vec(all(iszero, ηi; dims = 1))
        nonzero_cols = findall(!x -> x, zero_cols)
        if isempty(nonzero_cols)
            continue  # Skip if all columns are zero
        end

        # Use views to avoid copying ηi_nonzero
        ηi_nonzero = ηi[:, nonzero_cols]

        # Compute ηηi_sub = ηi_nonzero' * ηi_nonzero / T
        ηηi_sub = zeros(length(nonzero_cols), length(nonzero_cols))
        BLAS.syrk!('U', 'T', 1.0 / T, ηi_nonzero, 0.0, ηηi_sub)

        # Wrap ηηi_sub with Symmetric to represent the full symmetric matrix
        ηηi_sub_sym = Symmetric(ηηi_sub, :U)

        # Map back to original dimensions
        full_ηηi = zeros(Nmom, Nmom)
        full_ηηi[nonzero_cols, nonzero_cols] .= ηηi_sub_sym

        # Update per-thread accumulators
        breads[threadid] .+= full_ηηi
        meats[threadid] .+= full_ηηi .* σu²
    end

    # Sum over all threads
    bread = sum(breads)
    meat = sum(meats)

    # Compute the covariance matrix using Symmetric to exploit symmetry
    bread_sym = Symmetric(bread)
    bread_inv = inv(bread_sym)
    vcov_ols = bread_inv * meat * bread_inv / T
    return vcov_ols
end