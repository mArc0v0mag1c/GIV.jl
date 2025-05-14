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

Code optimized for performance with dynamic task distribution via @spawn.
"""
function solve_ols_vcov(umat, ηts)
    N, T, Nmom = size(ηts)

    # Preallocate the final matrices
    bread = zeros(Nmom, Nmom)
    meat = zeros(Nmom, Nmom)

    # Process a chunk of entities
    function process_chunk(chunk_range)::Tuple{Matrix{Float64},Matrix{Float64}}
        local_bread = zeros(Nmom, Nmom)
        local_meat = zeros(Nmom, Nmom)

        for i in chunk_range
            σu² = var(@view umat[i, :]; mean=zero(eltype(umat)))

            # Skip computations if variance is zero
            if σu² == 0
                continue
            end

            # Use views to avoid copying data
            @views ηi = ηts[i, :, :]

            # Identify non-zero columns to exploit sparsity
            zero_cols = vec(all(iszero, ηi; dims=1))
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

            # Update local accumulators
            local_bread .+= full_ηηi
            local_meat .+= full_ηηi .* σu²
        end

        return local_bread, local_meat
    end

    # Create smaller chunks for better load balancing
    nthreads = Threads.nthreads()
    # Use more chunks than threads for better load balancing
    n_chunks = nthreads * 2
    chunk_size = cld(N, n_chunks)

    # Create and spawn tasks with type annotation
    tasks = Vector{Task}()
    for c in 1:n_chunks
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, N)
        if start_idx <= end_idx
            task = Threads.@spawn process_chunk(start_idx:end_idx)
            push!(tasks, task)
        end
    end

    # Collect results from all tasks
    for task in tasks
        thread_bread, thread_meat = fetch(task)::Tuple{Matrix{Float64},Matrix{Float64}}
        bread .+= thread_bread
        meat .+= thread_meat
    end

    # Compute the covariance matrix using Symmetric to exploit symmetry
    bread_sym = Symmetric(bread)
    bread_inv = inv(bread_sym)
    vcov_ols = bread_inv * meat * bread_inv / T
    return vcov_ols
end