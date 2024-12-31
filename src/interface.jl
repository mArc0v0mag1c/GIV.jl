
"""
    giv(df, formula, id, t, weight; <keyword arguments>)


Estimate the GIV model. It returns a `GIVModel` object containing the estimated coefficients, standard errors, and other information.


# Arguments

- `df::DataFrame`: A DataFrame containing the data. Only balanced panel is supported for now. 
    It is recommended to sort the data by `t` and `id`.
- `formula::FormulaTerm`: A formula specifying the model. The formula should be in the form of
    `q + (C1 + C2+...) & endog(p) ~ exog_controls`, where 
        
    - `q` is the response variable, 
    - `endog(p)` indicates p is the endogenous variable,
    - `C1, C2, ...` can be the categorical variables to specify heterogeneous loadings, or exogenous variables to be interacted with the endogenous variable; when `x` are ommited, different entities are assumed to have the same loadings.
    - `exog_controls` are the exogenous variables. Notice that by default the model does not include an intercept term. If the mean is not zero, it is recommended to an entity fixed effect to demean the data.

    For example, `formula` can be written as
    ```julia
    @formula(q + id & endog(p) + C & endog(p) ~ id & η + id)
    ```


    Also notice that 
    - Endogenous variables are assumed to be on the left-hand side of the formula; 
    - All categorical&Bool variables are treated as fixed effects. 
- `id::Symbol`: The column name of the entity identifier.
- `t::Symbol`: The column name of the time identifier. `t` and `id` should uniquely identify each observation.
- `weight::Union{Symbol,Nothing}`: The column name of the weight variable for each entities. The weight must be non-negative. 
    You can flip swap the sign of `q` and `weight` if necessary. 

## Keyword Arguments

- `guess`: Initial guess for the coefficients in front of endogenous terms. If not provided, the initial guess is set using OLS. 
    Guess can be supplied in multiple ways:
    - A vector in the order of coefficient enters the formula. For categorical variables the order is determined by the variable.
    - A dictionary with the key being the name (either a string or a symbol) of the interaction term and the value being the initial guess 
    (a vector in the case of categorical variables and a number otherwise). In the example above, the initial guess can be provided as
    ```julia
    guess = Dict(:id => [1.0, 2.0], :η => 0.5)
- `algorithm::Symbol = :iv`: The algorithm to use for estimation. The default is `:iv`. The options are
    - `:iv`: The most flexible algorithm. It uses the moment condition such that E[u_i u_{S,-i}] = 0
    - `:iv_vcov`: `:iv_vcov` uses the same set of moment conditions as `:iv` but precomputes the covariance matrices 
    once and use them during the iteration. Constant weighting across time is required. 
    - `:debiased_ols`: `:debiased_ols` uses the moment condition such that E[u_i C_it p_it] = 1/ζ_St σ_i^2. ]
    It requires the adding-up constraint is satisifed so that Σ_i (q_it weight_i) = 0. 
    If not, the aggregate elasticity will be underestimated.
    - `:scalar_search`: `:scalar_search` uses the same moment condition `up` but requires the aggregate elasticity be constant across time. 
    It searches for the scalar of the aggregate elasticity and hence very efficient. 
    It can be used for diagnoises or forming initial guess for other algorithms. 
- `solver_options`: Options to pass to the `nlsolve`. 
- `quiet::Bool = false`: If `true`, suppress warnings and information.

  - `savedf::Bool = true`: If `true`, the input dataframe is saved in the field `df` of the returned `GIVModel` object. By default, it is `true`. For large datasets or repeated estimation, it is recommended to set it to `false`.


# Output

The output is `m::GIVModel`. Several important fields are:

  - `coef`: The estimated coefficients in front of the endogenous terms.
  - `vcov`: The estimated covariance matrix of the `coef`.
  - `factor_coef`: The estimated factor coefficients.
  - `agg_coef`: The estimated aggregate elasticity by `t`. If it is constant across `t`, it is stored as a scalar.
  - `residual_variance`: The estimated variance of the residual for each `id`.
  - `coefdf::DataFrame`: A `DataFrame`` containing the estimated coefficients.
  - `df::DataFrame`: A dataframe contains data used for estimation and the estimates.

"""
function giv(
    df,
    formula::FormulaTerm,
    id::Symbol,
    t::Symbol,
    weight::Union{Symbol,Nothing} = nothing;
    guess = nothing,
    exclude_pairs = Pair[],
    algorithm=:iv,
    solver_options = (;),
    quiet = false,
    savedf = true,
    return_vcov = true,
)
    formula = replace_function_term(formula) # FunctionTerm is inconvenient for saving&loading across Module
    df = preprocess_dataframe(df, formula, id, t, weight; quiet = quiet)
    slope_coefnames, factor_coefnames, endog_name, response_name = get_coefnames(df, formula)

    if length(exclude_pairs) > 0 && solver_options in [:debiased_ols, :scalar_search]
        @error("Moment exclusion not supported for the selected algorithm. 
        Proceed without `exclude_pairs`")
    end
    qmat, pmat, Cts, Cpts, ηts, Smat, exclmat = generate_matrices(
        df,
        formula,
        id,
        t,
        weight;
        algorithm = algorithm,
        quiet = quiet,
        exclude_pairs = exclude_pairs,
    )
    N, T, Nmom = size(Cts)

    # residualize against η
    η = reshape(ηts, N * T, :)
    cholη = cholesky(η' * η)
    uq, λq_endog = residualize_on_η(vec(qmat), η, cholη)
    uqmat = reshape(uq, N, T)
    uCp, λCp = residualize_on_η(reshape(Cpts, N * T, Nmom), η, cholη)
    uCpts = reshape(uCp, N, T, Nmom)

    guessvec = parse_guess(formula, guess, Val{algorithm}())
    ζ̂, converged = estimate_giv(
        uqmat,
        uCpts,
        Cts,
        Smat,
        exclmat,
        Val{algorithm}();
        guess = guessvec,
        quiet = quiet,
        solver_options = solver_options,
    )

    λ = λq_endog + λCp * ζ̂

    ûmat = uqmat + dropdims(sum(uCpts .* reshape(ζ̂, 1, 1, length(ζ̂)); dims = 3); dims = 3)
    if return_vcov
        σu²vec, Σζ = solve_vcov(ζ̂, ûmat, Smat, Cts)
        ols_vcov = solve_ols_vcov(ûmat, ηts)
        Σλ = ols_vcov + λCp * Σζ * λCp'
    else
        σu²vec, Σζ, Σλ = NaN * zeros(N), NaN * zeros(Nmom, Nmom), NaN * zeros(Nmom, Nmom)
    end
    coefdf = create_coef_dataframe(df, formula, [ζ̂; λ], [slope_coefnames; factor_coefnames], id)

    ζS = solve_aggregate_elasticity(ζ̂, Cts, Smat)
    ζS = length(unique(ζS)) == 1 ? ζS[1] : ζS

    dof = length(ζ̂) + length(λ)
    dof_residual = N * T - dof

    if savedf
        df = innerjoin(df, coefdf; on = intersect(names(df), names(coefdf)))

        aggdf = select(df, t) |> unique
        aggsym = Symbol(endog_name, "coef_agg")
        if string(aggsym) in names(df)
            throw(ArgumentError("The column name $(aggsym) already exists in the dataframe. Please rename it to avoid conflict."))
        end
        aggdf[!, Symbol(endog_name, "coef_agg")] .= ζS
        df = innerjoin(df, aggdf; on = t)
        sort!(df, [t, id])
        ressym = Symbol(response_name, "_residual")
        if string(ressym) in names(df)
            throw(ArgumentError("The column name $(ressym) already exists in the dataframe. Please rename it to avoid conflict."))
        end
        df[!, Symbol(response_name, "_residual")] = vec(ûmat)

        meansym = Symbol(response_name, "_mean")
        if string(meansym) in names(df)
            throw(ArgumentError("The column name $(meansym) already exists in the dataframe. Please rename it to avoid conflict."))
        end
    else
        df = nothing
    end

    return GIVModel(
        ζ̂,
        Σζ,
        λ,
        Σλ,
        λCp,
        σu²vec,
        ζS,
        formula,
        response_name,
        endog_name,
        slope_coefnames,
        factor_coefnames,
        id,
        t,
        weight,
        exclude_pairs,
        coefdf,
        df,
        converged,
        N,
        T,
        N * T,
        dof,
        dof_residual,
    )
end

function get_coefnames(df, formula)
    dfschema = schema(df, hint_for_categorical(df))
    response_term, slope_terms, endog_term, exog_terms = parse_endog(formula)
    slope_coefnames = apply_schema(slope_terms, FullRank(dfschema), GIVModel) |> coefnames
    if isa(slope_coefnames, String)
        slope_coefnames = [slope_coefnames]
    end
    slope_coefnames = replace(slope_coefnames, "(Intercept)" => "Constant")

    factor_coefnames = apply_schema(exog_terms, FullRank(dfschema), GIVModel) |> coefnames
    if isa(factor_coefnames, String)
        factor_coefnames = [factor_coefnames]
    end
    endog_name = apply_schema(endog_term, FullRank(dfschema), GIVModel) |> coefnames
    response_name = apply_schema(response_term, FullRank(dfschema), GIVModel) |> coefnames
    return slope_coefnames, factor_coefnames, endog_name, response_name
end

function preprocess_dataframe(df, formula, id, t, weight; quiet = false)

    # check data compatibility
    allvars = StatsModels.termvars(formula)
    df = select(df, unique([id, t, weight, allvars...])) |> disallowmissing!

    if any(nonunique(df, [id, t]))
        throw(ArgumentError("Observations are not uniquely identified by `id` and `t`"))
    end

    N, T = length(unique(df[!, id])), length(unique(df[!, t]))
    nrow(df) == N * T || throw(ArgumentError("Only balanced panel is supported for now"))
    all(df[!, weight] .>= 0) ||
        throw(ArgumentError("Weight must be non-negative. You can swap the sign of y and S if necessary."))
    sort!(df, [t, id])
    return df
end

function generate_matrices(
    df,
    formula,
    id,
    t,
    weight;
    algorithm=:iv,
    quiet = false,
    exclude_pairs = Pair[],
)
    # check data compatibility
    dfschema = StatsModels.schema(df, hint_for_categorical(df))
    N, T = length(unique(df[!, id])), length(unique(df[!, t]))
    (issorted(df, [t, id]) && nrow(df) == N * T) ||
        throw(ArgumentError("Only balanced panel is supported for now"))
    response_term, slope_terms, endog_term, exog_terms = parse_endog(formula)
    q = modelcols(apply_schema(response_term, dfschema, GIVModel), df)
    qmat = reshape(q, N, T)
    p = modelcols(apply_schema(endog_term, dfschema, GIVModel), df)
    pmat = reshape(p, N, T)
    if norm(var(pmat; dims = 1)) > eps(eltype(pmat))
        throw(ArgumentError("Price has to be constant across entities"))
    end
    # form a dictionary mapping from id to ind:
    idvec = df[1:N, id]
    id2ind = Dict(idvec[i] => i for i in 1:N)
    exclmat = zeros(Bool, N, N)
    for (id1, id2vec) in exclude_pairs
        for id2 in id2vec
            exclmat[id2ind[id1], id2ind[id2]] = true
            exclmat[id2ind[id2], id2ind[id1]] = true
        end
    end
    exclmat = BitArray(exclmat)

    slope_terms = MatrixTerm(apply_schema(slope_terms, FullRank(dfschema), GIVModel))
    C = modelcols(slope_terms, df)

    Nmom = size(C, 2)
    Cts = reshape(C, N, T, Nmom)
    if all(x -> iszero(x) || isone(x), Cts)
        Cts = BitArray(Cts)
    end
    Cpts = Cts .* pmat
    exog_terms = apply_schema(exog_terms, FullRank(dfschema), GIVModel) |> MatrixTerm
    η = modelcols(exog_terms, df)
    ηts = reshape(η, N, T, size(η, 2))

    S = df[!, weight]
    Smat = reshape(S, N, T)

    if !quiet &&
       algorithm ∈ [:scalar_search, :debiased_ols] &&
       any(>(eps(eltype(qmat))), sum(qmat .* Smat; dims = 1) .^ 2)
        @warn ("Adding-up constraints not satisfied. `up` and `scalar_search` algorithms may be biased.")
    end

    return qmat, pmat, Cts, Cpts, ηts, Smat, exclmat
end

parse_guess(formula, guess::Union{Nothing,Vector}, ::Val) = guess

function parse_guess(formula, guess::Dict, ::Val{:scalar_search})
    if "Aggregate" ∉ keys(guess)
        throw(ArgumentError("To use the scalar-search algorithm, specify the initial guess using \"Aggregate\" as the key in `guess`"))
    end
    return guess["Aggregate"]
end

function parse_guess(formula, guess::Dict, ::Any)
    guess = Dict(Symbol(x) => y for (x, y) in guess)
    response_term, slope_terms, endog_term, exog_terms = parse_endog(formula)
    return mapreduce(vcat, slope_terms) do term
        termsym = Symbol(term)
        if termsym == Symbol("1")
            termsym = :Constant
        end
        if string(termsym) ∉ string.(keys(guess))
            throw(ArgumentError("Initial guess for \"$(termsym)\" is missing"))
        else
            isa(guess[termsym], Number) ? [guess[termsym]] : guess[termsym]
        end
    end
end

function create_coef_dataframe(df, formula, coef, coefname, id)
    disallowmissing!(df, [Symbol(id); StatsModels.termvars(formula)])
    f_endog = FormulaTerm(
        ConstantTerm(0),
        Tuple(term for term in eachterm(formula.lhs) if has_endog(term)),
    )
    dfschema = StatsModels.schema(df, hint_for_categorical(df))
    slope_terms = eachterm(apply_schema(f_endog.rhs, FullRank(dfschema), GIVModel))
    exog_terms = eachterm(apply_schema(formula.rhs, FullRank(dfschema), GIVModel))
    terms = [slope_terms..., exog_terms...]
    categorical_terms_symbol = Symbol[]
    cat_symbol(t::CategoricalTerm) = [Symbol(t)]
    cat_symbol(t::InteractionTerm) = [Symbol(x) for x in t.terms if x isa CategoricalTerm]
    categorical_terms_symbol = [cat_symbol(t) for t in terms if has_categorical(t)]
    categorical_terms_symbol = unique(vcat(categorical_terms_symbol...))
    if categorical_terms_symbol == []
        categorical_terms_symbol = [id]
    end

    categories = select(df, categorical_terms_symbol) |> unique
    i = 1
    for term in terms
        termsym = Symbol(term)
        coefsym = Symbol(termsym, :_coef)
        coefnamecol = Symbol(termsym, :_coefname)
        if has_categorical(term)
            if term isa InteractionTerm
                term = InteractionTerm((x for x in term.terms if x isa CategoricalTerm))
            end
            catmat = modelcols(term, categories)
            termcoef = coef[i:i+size(catmat)[2]-1]
            termcoefname = coefname[i:i+size(catmat)[2]-1]
            termmat = modelcols(term, categories)
            categories[!, coefsym] = termmat * termcoef
            categories[!, coefnamecol] = termcoefname[findfirst.(eachrow(Bool.(termmat)))]
            i += size(catmat)[2]
        elseif term isa InterceptTerm{false}
            continue
        else
            categories[!, coefsym] .= coef[i]
            categories[!, coefnamecol] .= coefname[i]
            i += 1
        end
    end
    if i != length(coef) + 1
        println(length(coef))
        throw(ArgumentError("Number of coefficients does not match the number of terms. You may be using different formula or dataframe for estimation and creating coef dataframe."))
    end

    return categories
end

"""
    predict_endog(m::GIVModel; <keyword arguments>)

Predict the endogenous variable based on the estimated GIV model.

# Arguments

- `m::GIVModel`: The estimated GIV model.
- `df::DataFrame`: The dataframe to predict the endogenous variable. 
    It uses the saved dataframe in `m` by default. A different dataframe can be used for prediction to construct counterfactuals.

## Keyword Arguments
- `coef::Vector{Float64}`: The estimated coefficients. 
    It uses the coefficients in `m` by default. A different set of coefficients can be
    used to construct counterfactuals.
- `factor_coef::Vector{Float64}`: The estimated factor coefficients.
    It uses the factor coefficients in `m` by default. A different set of factor coefficients can be
    used to construct counterfactuals.
- `residual::Symbol`: The name of the residual column in the dataframe. 
    By default it uses the estimated residual; different residuals can be used to construct counterfactuals.
- `formula::FormulaTerm`: The formula used to estimate the model.
- `id::Symbol`: The name of the entity identifier.
- `t::Symbol`: The name of the time identifier.
- `weight::Symbol`: The name of the weight variable.
"""
function predict_endog(
    m::GIVModel,
    df = m.df;
    coef = m.coef,
    factor_coef = m.factor_coef,
    residual = Symbol(m.responsename, "_residual"),
    formula = m.formula,
    id = m.idvar,
    t = m.tvar,
    weight = m.weightvar,
    quiet = false,
)
    if isnothing(df)
        throw(ArgumentError("DataFrame not saved. Rerun the model with `savedf = true`"))
    end
    # add residual to the rhs
    formula = FormulaTerm(formula.lhs, tuple(eachterm(formula.rhs)..., Term(residual)))
    factor_coef = [factor_coef; one(eltype(factor_coef))]
    df = preprocess_dataframe(df, formula, id, t, weight)
    matricies = generate_matrices(df, formula, id, t, weight)
    qmat, pmat, Cts, Cpts, ηts, Smat = matricies
    N, T, Nmom = size(Cts)
    mkc_err = sum(qmat .* Smat; dims = 1) .^ 2
    if !quiet && any(>(eps(eltype(qmat))), mkc_err)
        @warn ("Adding-up constraints not satisfied. Interpret the results with caution.")
    end
    aggcoef = solve_aggregate_elasticity(coef, Cts, Smat)
    λη = reshape(ηts, N * T, :) * factor_coef
    netshock = reshape(λη, N, T)
    shockS = sum(netshock .* Smat; dims = 1) |> vec
    pvec = shockS ./ aggcoef
    return pvec
end